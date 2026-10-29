"""Predeclared local reference bank/refinements, original64rough initial fields."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing
from pathlib import Path
import time
import traceback

import jax
import numpy as np
from ml_collections import ConfigDict
from scipy import sparse

from examples.corrosion2d.residual_experiment import grid, reference_spec
from examples.corrosion2d.local_reference import negative_laplacian
from examples.electropolish2d.operator_data import condition_config
from examples.electropolish2d.operator_reference import solve


BASE = Path('/root/autodl-tmp/data/electropolish2d/operator_test64_20261028')
ROOT = Path('/root/autodl-tmp/data/electropolish2d/operator_local_test64_20261029')
RUN = Path('/root/autodl-tmp/tf-logs/electropolish2d/20261028-electro-operator-online-global-adam-01')
TIMES = np.unique(np.r_[np.linspace(0., 1., 21), [.001, .003, .01, .025]])
REFINEMENT = (0, 21, 42, 63)


def reference(index, nx, native, resume):
    c = ConfigDict(json.loads((RUN/'config.json').read_text()))
    manifest = json.loads((BASE/'manifest.json').read_text())
    c = condition_config(c, manifest['surface_ids'][index], '')
    c.sample.Nx, c.sample.Ny = nx, (nx+1)//2
    xx, yy, points = grid(c)
    name = 'native101' if native else f'local{nx}'
    path = ROOT/f'condition_{index:03d}/{name}.npz'
    spec = reference_spec(c)
    if not native:
        spec['operator'] = 'local negative Laplacian; centered FD with side Neumann mirror'
        spec['other_boundary'] = 'left/right: phi_n=q_n=0'
        del spec['delta'], spec['kernel']
    e = c.experiment
    if path.exists():
        if not resume:
            raise FileExistsError(path)
        with np.load(path) as stored:
            metadata = json.loads(str(stored['metadata_json']))
            np.testing.assert_array_equal(stored['time'], TIMES)
            fields = stored['fields']
            assert fields.shape == (len(TIMES), c.sample.Ny, nx, 2)
            assert np.isfinite(fields).all()
            assert fields[..., 0].min() >= -1e-5 and fields[..., 0].max() <= 1.00001
        assert metadata['spec'] == spec and metadata['surface_id'] == int(c.surface.seed)
        assert metadata['kind'] == name and metadata['index'] == index
        assert metadata['maximum_time_refinement_error'] < e.reference_error_tolerance
        return dict(path=str(path), reused=True, **metadata)
    if native:
        with np.load(BASE/f'condition_{index:03d}/reference.npz') as stored:
            i, j, w = stored['graph_row'], stored['graph_col'], stored['graph_weight'].astype(float)
        matrix = sparse.diags(np.bincount(i, weights=w, minlength=len(points)))-sparse.csr_matrix((w, (i, j)), shape=(len(points), len(points)))
    else:
        matrix = negative_laplacian(nx, c.sample.Ny)
    coarse, coarse_info = solve(c, matrix.tocsr(), points, TIMES, e.reference_rtol, e.reference_atol, e.reference_max_step)
    fine, fine_info = solve(c, matrix.tocsr(), points, TIMES, e.refined_rtol, e.refined_atol, e.refined_max_step)
    errors = np.linalg.norm(fine-coarse, axis=1)/np.linalg.norm(fine, axis=1)
    maximum = float(errors.max())
    if maximum >= e.reference_error_tolerance:
        raise ValueError(f'Unconverged time reference {index}/{nx}/{native}: {maximum}')
    path.parent.mkdir(parents=True, exist_ok=True)
    shape = (len(TIMES), c.sample.Ny, nx, 2)
    metadata = dict(index=index, surface_id=int(c.surface.seed), nx=nx, ny=c.sample.Ny,
                    kind=name, spec=spec, coarse=coarse_info, fine=fine_info,
                    maximum_time_refinement_error=maximum,
                    boundary='Top0/bottom1 physical phi,c; side phi_n=q_n=0' if not native else 'Recorded in-domain native pair graph',
                    original_initial_condition=True)
    with path.open('xb') as stream:
        np.savez_compressed(stream, time=TIMES, physical_time=TIMES*c.nondim.T_c,
                            X=xx, Y=yy, fields=fine.reshape(shape),
                            coarse=coarse.reshape(shape), metadata_json=np.array(json.dumps(metadata)))
    return dict(path=str(path), **metadata)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true', help='Explicitly validate and reuse completed references')
    args = parser.parse_args()
    assert jax.default_backend() == 'cpu' and jax.config.x64_enabled
    if not args.resume:
        ROOT.mkdir(parents=True, exist_ok=False)
    jobs = [(i, 101, False) for i in range(64)]
    jobs += [(i, nx, False) for nx in (201, 401) for i in REFINEMENT]
    jobs += [(i, 101, True) for i in REFINEMENT]
    manifest = dict(base=str(BASE), config=str(RUN/'config.json'), times=TIMES.tolist(),
                    refinement_indices=REFINEMENT, jobs=jobs, workers=4,
                    selection='Predeclared before new training/predictions; original endpoint slopes preserved')
    if args.resume:
        stored = json.loads((ROOT/'manifest.json').read_text())
        assert stored == json.loads(json.dumps(manifest))
    else:
        (ROOT/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    start = time.perf_counter()
    records, failures = [], []
    with ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context('spawn')) as pool:
        futures = {pool.submit(reference, *job, args.resume): job for job in jobs}
        for future in as_completed(futures):
            try:
                result = future.result()
            except Exception:
                failure = dict(job=futures[future], traceback=traceback.format_exc())
                failures.append(failure)
                print('REFERENCE FAILED', json.dumps(failure), flush=True)
                continue
            records.append(result)
            print('REFERENCE', result['kind'], result['index'], result['fine']['seconds'],
                  result['maximum_time_refinement_error'], flush=True)
    summary = dict(wall_seconds=time.perf_counter()-start, records=records,
                   failures=failures,
                   max_time_refinement_error=max(r['maximum_time_refinement_error'] for r in records))
    (ROOT/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    if failures:
        raise RuntimeError(f'{len(failures)}reference jobs failed; see explicit failures in summary.json')
    print('COMPLETE', summary['wall_seconds'], summary['max_time_refinement_error'], flush=True)
