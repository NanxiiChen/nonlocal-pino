"""Extend the native bank to64test ellipsoids with unchanged CPU RK4 physics."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import importlib
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import time

import numpy as np


def write(path, value):
    temporary = path.with_suffix('.partial')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def reference_error(coarse, fine, tolerance):
    """Use full-volume snapshot norms, before selecting evaluation coordinates."""
    if (coarse.shape != fine.shape or fine.ndim != 4
            or not np.isfinite(coarse).all() or not np.isfinite(fine).all()
            or fine.min() < -1.000001 or fine.max() > 1.000001):
        raise ValueError('Invalid full-cube reference fields')
    den = np.linalg.norm(fine.reshape(len(fine), -1), axis=1)
    if np.any(den <= 0):
        raise ValueError('Reference snapshot has zero norm')
    errors = np.linalg.norm((fine-coarse).reshape(len(fine), -1), axis=1)/den
    if not np.isfinite(errors).all() or errors.max() >= tolerance:
        raise ValueError('Time-refinement gate failed: '+str(errors.tolist()))
    return errors


def solve_group(config, parameters, indices, cpus, output):
    # Pin before importing JAX, so each spawned CPU backend has bounded threads.
    os.sched_setaffinity(0, set(cpus))
    import jax
    import jax.numpy as jnp
    from ml_collections import ConfigDict
    from examples.icemelting3d.ellipsoids import signed_distance_numpy
    from examples.icemelting3d.operator_reference import make_integrator
    c = ConfigDict(config)
    if jax.default_backend() != 'cpu' or not jax.config.x64_enabled:
        raise ValueError('Reference workers require CPU and explicit FP64')
    folder = Path(output)
    n = c.sample.Nx
    axis = np.linspace(0, 1, n)
    points = np.stack(np.meshgrid(axis, axis, axis, indexing='ij'), -1).reshape(-1, 3)
    chosen = np.linspace(0, n-1, c.operator.evaluation_axis_points).round().astype(int)
    ids = np.ravel_multi_index(np.meshgrid(chosen, chosen, chosen, indexing='ij'), (n,n,n)).ravel()
    slices = np.arange(n**3).reshape(n,n,n)[:, :, n//2].ravel()
    coarse_fn = make_integrator(c, c.operator.reference_dt)
    fine_fn = make_integrator(c, c.operator.reference_dt/2)
    records = []
    for index in indices:
        start = time.perf_counter(); p = np.asarray(parameters[index])
        initial = np.tanh(signed_distance_numpy(points, p, c.initial.reference_sdf_iterations)
                          /(np.sqrt(2)*c.physical.eps_star)).reshape(n,n,n)
        coarse = np.asarray(coarse_fn(jnp.asarray(initial)))
        fine = np.asarray(fine_fn(jnp.asarray(initial)))
        errors = reference_error(coarse, fine, c.operator.reference_tolerance)
        info = dict(index=index, maximum_time_refinement_error=float(errors.max()),
            seconds=time.perf_counter()-start, phi_min=float(fine.min()), phi_max=float(fine.max()),
            dt=c.operator.reference_dt, refined_dt=c.operator.reference_dt/2,
            cpu_affinity=cpus, scope=f'Full{n}cube native graph time convergence, not local continuum accuracy')
        path = folder/f'condition_{index:03d}.npz'
        if path.exists():
            raise FileExistsError(path)
        np.savez_compressed(path, parameters=p, time=np.linspace(0, 1, len(fine)),
            points=points[ids], phi=fine.reshape(len(fine), -1)[:, ids],
            coarse=coarse.reshape(len(fine), -1)[:, ids], slice_points=points[slices],
            slice_phi=fine.reshape(len(fine), -1)[:, slices],
            solid_fraction=((fine+1)/2).mean(axis=(1,2,3)), positive_fraction=(fine>0).mean(axis=(1,2,3)),
            metadata_json=np.array(json.dumps(info, allow_nan=False)))
        records.append(info)
        print('REFERENCE64 '+json.dumps(info), flush=True)
    return records


def prepare(config, source, output, workers, cores_per_worker, extension_seed):
    allowed = sorted(os.sched_getaffinity(0))
    if (workers < 1 or cores_per_worker < 1 or workers*cores_per_worker > len(allowed)
            or not 0 <= extension_seed < 2**32):
        raise ValueError('Explicit CPU worker/affinity budget and uint32 extension seed required')
    os.sched_setaffinity(0, {allowed[0]})
    import jax
    from examples.icemelting3d.ellipsoids import sample_parameters, validate_parameters
    from examples.icemelting3d.operator_data import identity, load_dataset, validate_config
    if jax.default_backend() != 'cpu' or not jax.config.x64_enabled:
        raise ValueError('Reference preparation requires CPU FP64')
    c = importlib.import_module(config).get_config()
    validate_config(c)
    if c.operator.monitor_conditions != 2 or c.operator.test_conditions != 64 or Path(c.operator.dataset_path) != output:
        raise ValueError('Explicit2monitor+64test destination configuration required')
    old = c.copy_and_resolve_references()
    old.operator.dataset_path = str(source); old.operator.test_conditions = 8
    original, refs = load_dataset(old)
    if original['monitor_indices'] != [0,1] or original['test_indices'] != list(range(2,10)):
        raise ValueError('Expected original10inputs, including independently checked case005')
    parameters = np.concatenate((np.asarray(original['parameters']), np.asarray(
        sample_parameters(jax.random.key(extension_seed), 56, c.initial), dtype=float)))
    validate_parameters(parameters)
    if parameters.shape != (66,10) or len(set(map(tuple, parameters.tolist()))) != 66:
        raise ValueError('Distinct finite66ellipsoid inputs required')
    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(status='preparing', identity=identity(c), parameters=parameters.tolist(),
        monitor_indices=[0,1], test_indices=list(range(2,66)), grid_order=original['grid_order'],
        selection='Preserve original10inputs; append56declared independent inputs before training',
        original_source=str(source), extension_seed=extension_seed, new_indices=list(range(10,66)),
        reference_scope='Native nonlocal graph, independent CPU solver; not local continuum reference')
    write(output/'manifest.json', manifest)
    records = []
    for i in range(10):
        path = source/f'condition_{i:03d}.npz'; target = output/path.name
        shutil.copyfile(path, target)
        if path.read_bytes() != target.read_bytes():
            raise ValueError('Original reference copy changed')
        with np.load(path) as data:
            records.append(dict(index=i, copied_from=str(path), **json.loads(str(data['metadata_json']))))
    started = time.perf_counter()
    payload = json.loads(c.to_json())
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as executor:
        jobs = [executor.submit(solve_group, payload, parameters.tolist(), list(range(10+k,66,workers)),
                    allowed[k*cores_per_worker:(k+1)*cores_per_worker], str(output)) for k in range(workers)]
        for job in jobs:
            records.extend(job.result())
    records.sort(key=lambda r:r['index'])
    # Run the ordinary strict physical-grid/IC/reference loader before success.
    loaded, complete = load_dataset(c)
    if len(complete) != 66 or loaded['test_indices'] != list(range(2,66)):
        raise ValueError('Incomplete extended reference bank')
    manifest['status'] = 'passed'; write(output/'manifest.json', manifest)
    write(output/'summary.json', dict(status='passed', records=records,
        reference_workers=workers, cores_per_worker=cores_per_worker,
        extension_seconds=time.perf_counter()-started, original_reference_count=len(refs),
        new_reference_count=56, test_count=64, created_utc=datetime.now(timezone.utc).isoformat()))
    print('REFERENCE64_COMPLETE '+str(output), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True); p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True); p.add_argument('--workers', type=int, required=True)
    p.add_argument('--cores-per-worker', type=int, required=True); p.add_argument('--extension-seed', type=int, required=True)
    args = p.parse_args()
    prepare(args.config, args.source, args.output, args.workers, args.cores_per_worker, args.extension_seed)
