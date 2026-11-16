"""Prepare nested known-input libraries only; no reference solves or training."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path
import time

import numpy as np

SIZES = (64, 128, 256, 512)
CERAMIC_INPUT = Path('/root/autodl-tmp/data/particle_corrosion2d/20261107-spaced-six-01/inputs64')
CERAMIC_REFERENCE = Path('/root/autodl-tmp/data/particle_corrosion2d/local_test64_20261111-01')
CERAMIC_DEV = Path('/root/autodl-tmp/data/particle_corrosion2d/20261107-spaced-six-01/local_dev4')
ELECTRO_INPUT = Path('/root/autodl-tmp/data/electropolish2d/operator_test64_20261028')


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def electro_coefficients(c, count, seed):
    import jax
    import jax.numpy as jnp
    from examples.electropolish2d.online_conditions import rough_surface
    x = jnp.asarray(np.linspace(0, 1, c.operator.online.curve_points).astype(np.float32))
    @jax.jit
    def candidates(key):
        u = jax.random.uniform(key, (32, c.surface.n_modes, 3), dtype=jnp.float32)
        p = jnp.stack(((.3+.7*u[..., 0])*c.surface.amplitude,
            c.surface.freq_lo+(c.surface.freq_hi-c.surface.freq_lo)*u[..., 1],
            2*jnp.pi*u[..., 2]), -1)
        y = jax.vmap(lambda a: rough_surface(x, a, c.surface.y0))(p)
        return p, (y.min(-1) > c.operator.online.height_min) & (y.max(-1) < c.operator.online.height_max)
    accepted = []
    for draw in range(4096):
        values, valid = candidates(jax.random.fold_in(jax.random.key(seed), draw))
        accepted.extend(np.asarray(values)[np.asarray(valid)])
        if len(accepted) >= count:
            return np.asarray(accepted[:count], dtype=np.float32), (draw+1)*32
    raise RuntimeError('Fixed-profile preparation exhausted4096candidate batches')


def prepare_electro(output, seed):
    from examples.electropolish2d.configs.train_operator_threeway import get_config
    from examples.electropolish2d.condition_library import load_library
    from ml_collections import ConfigDict
    c = get_config(); start = time.perf_counter()
    profiles, proposals = electro_coefficients(c, max(SIZES), seed)
    reference = read(ELECTRO_INPUT/'manifest.json')
    forbidden = [np.asarray(reference['coefficients'], dtype=np.float32)]
    with np.load(Path(c.operator.dataset_path)/'conditions.npz', allow_pickle=False) as data:
        for identity in data['surface_ids']:
            rng = np.random.RandomState(int(identity))
            forbidden.append(np.asarray([[(rng.uniform(.3, 1)*c.surface.amplitude,
                rng.uniform(c.surface.freq_lo, c.surface.freq_hi), rng.uniform(0, 2*np.pi))
                for _ in range(c.surface.n_modes)]], dtype=np.float32))
    forbidden = np.concatenate(forbidden)
    if np.any(np.max(np.abs(profiles[:, None]-forbidden[None]), axis=(2, 3)) < 1e-6):
        raise ValueError('Training profile overlaps a monitoring/test input')
    path = output/'electropolish'; path.mkdir()
    np.savez_compressed(path/'coefficients.npz', coefficients=profiles)
    write(path/'manifest.json', dict(schema='electropolish_initial_library_v1',
        config=c.to_dict(), count=len(profiles), seed=seed, candidate_count=proposals,
        evolved_fields_generated=False, pde_solves=0, reference_overlap=0,
        excluded_inputs=[str(ELECTRO_INPUT), c.operator.dataset_path],
        generation='Original FP32 uniform amplitude/frequency/phase prior and2000vertex height rejection',
        sizes=SIZES, seconds=time.perf_counter()-start))
    c.operator.condition_library = ConfigDict(dict(path=str(path), count=max(SIZES),
        sampling='uniform_with_replacement'))
    checked, _ = load_library(c)
    np.testing.assert_array_equal(checked, profiles)
    return dict(path=str(path), count=len(profiles), seconds=time.perf_counter()-start)


def ceramic_worker(arguments):
    from examples.particle_corrosion2d.online import build_input
    config, seed, identity = arguments
    return build_input(config, seed, identity, 0, None)


def prepare_ceramic(output, seed, workers):
    from examples.particle_corrosion2d.configs.train_fixed_geometry import get_config
    from examples.particle_corrosion2d.train import load_bank
    from examples.particle_corrosion2d.online import pad_initial_bank, geometry_identity
    c = get_config(); start = time.perf_counter()
    old, manifest = load_bank(CERAMIC_INPUT, c); old = pad_initial_bank(old, c)
    fixed = {name: np.empty((max(SIZES), *value.shape[1:]), dtype=value.dtype)
             for name, value in old.items()}
    for name in fixed: fixed[name][:64] = old[name]
    records = list(manifest['geometries']); metadata = list(manifest['records'])
    forbidden = {geometry_identity(case['geometry']) for root in (CERAMIC_REFERENCE, CERAMIC_DEV)
                 for case in read(root/'manifest.json')['cases']}
    identities = set()
    for geometry in records:
        identity = geometry_identity(geometry)
        if identity in identities or identity in forbidden: raise ValueError('Nonunique or test-overlapping original geometry')
        identities.add(identity)
    c.operator.geometry_count = max(SIZES)
    requests = [(c.to_dict(), seed, i) for i in range(64, max(SIZES))]
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        for index, (data, info) in enumerate(pool.map(ceramic_worker, requests), start=64):
            identity = geometry_identity(info['geometry'])
            if identity in identities or identity in forbidden: raise ValueError('Duplicate or test-overlapping new geometry')
            identities.add(identity)
            for name in fixed: fixed[name][index] = data[name]
            records.append(info['geometry']); metadata.append(info)
            if (index+1) % 64 == 0: print('PREPARED_CERAMIC',index+1,flush=True)
    roots = {}
    for count in SIZES:
        path = output/f'ceramic-{count}'; path.mkdir(); c.operator.geometry_count=count
        np.savez_compressed(path/'inputs.npz', **{name:value[:count] for name,value in fixed.items()})
        write(path/'manifest.json', dict(config=c.to_dict(), runtime_seed=seed,
            geometries=records[:count], records=metadata[:count],
            halo_size=c.operator.online.halo, neighbor_size=c.operator.online.neighbors,
            fixed_keys=manifest['fixed_keys'], graph_keys=manifest['graph_keys'],
            seconds=time.perf_counter()-start, training_fields_generated=False,
            reference_overlap=0, nested_prefix_count=count, original64=str(CERAMIC_INPUT),
            sampling='Nested fixed geometries; original64physical inputs retained; spatial refresh remains enabled'))
        roots[str(count)] = str(path)
    return dict(paths=roots, unique_geometries=len(identities), seconds=time.perf_counter()-start,
                memory_bytes=sum(v.nbytes for v in fixed.values()), pde_solves=0)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--workers',type=int,required=True);a=p.parse_args()
    if a.seed<0 or a.workers<1:raise ValueError('Nonnegative seed and positive workers required')
    output=a.output.resolve();output.mkdir(parents=True,exist_ok=False)
    result=dict(status='preparing',seed=a.seed,root=str(output),sizes=SIZES)
    write(output/'preparation.json',result)
    result['electropolish']=prepare_electro(output,a.seed)
    write(output/'preparation.json',result)
    result['ceramic']=prepare_ceramic(output,a.seed,a.workers)
    result['status']='passed';write(output/'preparation.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
