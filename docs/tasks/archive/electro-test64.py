"""Frozen operator test; CPU reference preparation and separate GPU evaluation.

Run from repository root with its .venv. All data/checkpoints use direct paths.
The test identities are fixed before any evolved field or prediction is seen.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing
from pathlib import Path
import time

import numpy as np


RUN = Path('/root/autodl-tmp/tf-logs/electropolish2d/20261028-electro-operator-online-global-adam-01')
ROOT = Path('/root/autodl-tmp/data/electropolish2d/operator_test64_20261028')
OUT = Path(__file__).resolve().parent


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def config():
    from ml_collections import ConfigDict
    return ConfigDict(json.loads((RUN / 'config.json').read_text()))


def worker(index, identity):
    from examples.electropolish2d.operator_data import condition_config
    from examples.corrosion2d.residual_experiment import generate_reference
    c = condition_config(config(), identity, ROOT / f'condition_{index:03d}/reference.npz')
    generate_reference(c)
    return index


def prepare():
    import jax
    from examples.electropolish2d.losses import _rough_surface
    c = config()
    assert jax.default_backend() == 'cpu' and jax.config.x64_enabled
    ROOT.mkdir(parents=True, exist_ok=False)
    old = np.load(Path(c.operator.dataset_path) / 'conditions.npz')['surface_ids']
    rng = np.random.default_rng(20261028)  # independent test provenance, not a tuned strategy
    chosen, coefficients = [], []
    curves = np.linspace(0, 1, c.operator.online.curve_points)
    attempts = 0
    while len(chosen) < 64:
        identity = int(rng.integers(0, 2**31))
        attempts += 1
        if identity in old or identity in chosen:
            continue
        source = np.random.RandomState(identity)
        modes = np.array([(source.uniform(.3, 1)*c.surface.amplitude,
                           source.uniform(c.surface.freq_lo, c.surface.freq_hi),
                           source.uniform(0, 2*np.pi)) for _ in range(c.surface.n_modes)])
        y = _rough_surface(curves, c.surface.y0, modes)
        if y.min() <= c.operator.online.height_min or y.max() >= c.operator.online.height_max:
            continue
        chosen.append(identity)
        coefficients.append(modes)
    coeff = np.stack(coefficients)
    actual = np.load(RUN / 'online_coefficients.npy').reshape(-1, c.surface.n_modes, 3)
    trained = {tuple(a.ravel()) for a in actual}
    assert all(tuple(a.astype(np.float32).ravel()) not in trained for a in coeff)
    manifest = dict(count=64, test_seed=20261028, surface_ids=chosen,
                    coefficients=coeff.tolist(), candidate_attempts=attempts,
                    checkpoint=str(RUN / 'model_final.eqx'), source_config=str(RUN / 'config.json'),
                    distribution='Original six independent mode triples, conditioned on online height bounds',
                    curve_points=c.operator.online.curve_points,
                    height_bounds=[c.operator.online.height_min, c.operator.online.height_max],
                    exact_training_profile_overlap=0, existing_bank_identity_overlap=0,
                    scope='Fresh post-training fixed test only; no checkpoint selection or continuum claim')
    write_json(ROOT / 'manifest.json', manifest)
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context('spawn')) as pool:
        futures = [pool.submit(worker, i, identity) for i, identity in enumerate(chosen)]
        for future in as_completed(futures):
            print('REFERENCE COMPLETE', future.result(), flush=True)
    refs = [json.loads((ROOT / f'condition_{i:03d}/reference.json').read_text()) for i in range(64)]
    summary = dict(wall_seconds=time.perf_counter()-start, workers=4, count=64,
                   max_time_refinement_error=max(r['refinement_max_relative_l2'] for r in refs),
                   sum_solver_seconds=sum(r[s]['seconds'] for r in refs for s in ('coarse', 'fine')))
    write_json(ROOT / 'preparation.json', summary)
    print('PREPARATION', json.dumps(summary), flush=True)


def evaluate():
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    from nonlocal_pinn.training.runtime import configure_gpu_policy
    from examples.electropolish2d.operator_training import _build_network, predict_fields
    from examples.electropolish2d.operator_data import condition_config
    from examples.corrosion2d.residual_experiment import (
        grid, boundary_values, initial_fields, load_reference, field_metrics)
    c = config()
    configure_gpu_policy(c.train.gpu_policy)
    jax.config.update('jax_enable_x64', False)
    jax.config.update('jax_default_matmul_precision', c.train.matmul_precision)
    assert jax.default_backend() == 'gpu' and c.arch.local_input == 'zero'
    manifest = json.loads((ROOT / 'manifest.json').read_text())
    assert manifest['count'] == 64
    old_bank = json.loads((Path(c.operator.dataset_path) / 'dataset.json').read_text())
    with np.load(old_bank['reference_paths'][0]) as stored:
        graph = {name: stored[name] for name in ('graph_row', 'graph_col', 'graph_weight')}
    template = _build_network(c.arch, c.operator.image_shape, jax.random.key(0))
    net = eqx.tree_deserialise_leaves(RUN / 'model_final.eqx', template)
    xx, yy, points = grid(c)
    _, height, width = c.operator.image_shape
    sx, sy = np.meshgrid(np.linspace(0, 1, width), np.linspace(0, .5, height), indexing='xy')
    sensors = np.column_stack([sx.ravel(), sy.ravel()])
    x = jnp.asarray(points, dtype=jnp.float32)
    times = jnp.linspace(0, 1, c.experiment.reference_snapshots)
    predict = eqx.filter_jit(lambda model, image, initial: predict_fields(model, image, x, times, initial, c))
    records, images, initial_all, predictions = [], [], [], []
    start = time.perf_counter()
    for i, identity in enumerate(manifest['surface_ids']):
        local = condition_config(c, identity, ROOT / f'condition_{i:03d}/reference.npz')
        ref = load_reference(local)
        for name, expected_graph in graph.items():
            np.testing.assert_array_equal(ref[name], expected_graph)
        initial = initial_fields(local, points)
        z, o, values = boundary_values(c, points)
        initial[z | o] = values[z | o]
        sensor_values = initial_fields(local, sensors)
        z, o, values = boundary_values(c, sensors)
        sensor_values[z | o] = values[z | o]
        image = sensor_values[:, 0].reshape(1, height, width).astype(np.float32)
        expected = np.stack([ref['phi'][0].ravel(), ref['c'][0].ravel()], -1)
        np.testing.assert_allclose(initial, expected, rtol=0, atol=3e-7)
        tick = time.perf_counter()
        pred = np.asarray(predict(net, jnp.asarray(image), jnp.asarray(initial, dtype=jnp.float32)))
        infer_seconds = time.perf_counter()-tick
        pred = pred.reshape(len(times), c.sample.Ny, c.sample.Nx, 2)
        assert np.isfinite(pred).all()
        m = field_metrics(pred, ref)
        target = np.stack([ref['phi'], ref['c']], -1)
        for channel, label in enumerate(('phi', 'c')):
            for prefix, a, b in (('pooled', pred[..., channel], target[..., channel]),
                                  ('terminal', pred[-1, ..., channel], target[-1, ..., channel])):
                independent = float(np.linalg.norm(a-b)/np.linalg.norm(b))
                np.testing.assert_allclose(m[prefix+'_rel_l2_'+label], independent, rtol=2e-6)
        records.append(dict(index=i, surface_id=identity, inference_seconds=infer_seconds, **m))
        images.append(image)
        initial_all.append(initial.astype(np.float32))
        predictions.append(pred)
        print('EVALUATED', i, m['pooled_rel_l2_phi'], m['terminal_rel_l2_phi'], flush=True)
    stats = {}
    bootstrap = np.random.default_rng(6400).integers(0, 64, (4000, 64))
    for name in ('pooled_rel_l2_phi', 'terminal_rel_l2_phi', 'pooled_rel_l2_c', 'terminal_rel_l2_c'):
        a = np.array([r[name] for r in records])
        stats[name] = dict(mean=float(a.mean()), median=float(np.median(a)),
                           p90=float(np.quantile(a, .9)), maximum=float(a.max()),
                           std=float(a.std(ddof=1)), worst_index=int(a.argmax()),
                           bootstrap_mean_95ci=np.quantile(a[bootstrap].mean(1), [.025, .975]).tolist())
    result = dict(checkpoint=manifest['checkpoint'], count=64, final_training_step=5000,
                   statistics=stats, records=records, wall_seconds=time.perf_counter()-start,
                   warm_inference_mean_seconds=float(np.mean([r['inference_seconds'] for r in records[1:]])),
                   first_inference_seconds=records[0]['inference_seconds'],
                   preparation=json.loads((ROOT / 'preparation.json').read_text()),
                   scope=manifest['scope'])
    write_json(ROOT / 'evaluation.json', result)
    write_json(OUT / 'electro-test64-results.json', result)
    write_json(OUT / 'electro-test64-manifest.json', manifest)
    np.savez_compressed(ROOT / 'prediction.npz', X=xx, Y=yy, time=np.asarray(times),
                        prediction=np.stack(predictions), images=np.stack(images),
                        initial=np.stack(initial_all))
    print('RESULT', json.dumps(stats), flush=True)
    print('EVALUATION_SECONDS', result['wall_seconds'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('prepare', 'evaluate'))
    args = parser.parse_args()
    (prepare if args.action == 'prepare' else evaluate)()
