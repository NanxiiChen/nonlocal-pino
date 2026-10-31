"""CPU-only audits of cached inputs, actual draws and checkpoint fields."""
import argparse
import ast
import json
from pathlib import Path
import time

import numpy as np


REPO = Path('/root/nonlocal-pinn')
ROOT = Path('/root/autodl-tmp/tf-logs/electropolish2d')
PREFIX = '20261031-electro-spatial-01'
BASE = Path('/root/autodl-tmp/data/electropolish2d/operator_test64_20261028')
LOCAL = Path('/root/autodl-tmp/data/electropolish2d/operator_local_test64_20261029')
MODES = ('full', 'fixed_pool', 'rolling_pool')


def write(name, result):
    (REPO/'docs/tasks'/name).write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')


def inputs():
    from examples.electropolish2d.configs.train_operator import get_config
    from examples.electropolish2d.losses import _rough_surface, _sdf_to_curve
    c = get_config()
    manifest = json.loads((BASE/'manifest.json').read_text())
    with np.load(BASE/'prediction.npz') as f:
        images, initial, xx, yy = (f[k] for k in ('images', 'initial', 'X', 'Y'))
    _, h, w = c.operator.image_shape
    sx, sy = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, .5, h), indexing='xy')
    sensors = np.column_stack([sx.ravel(), sy.ravel()])
    curve_x = np.linspace(0, 1, 2000)
    k = np.sqrt(c.physical.omega_phi/(2*c.physical.alpha_phi))*c.nondim.L_c
    image_gap, initial_gap, reference_gate = [], [], []
    for i, modes in enumerate(manifest['coefficients']):
        curve_y = _rough_surface(curve_x, c.surface.y0, modes)
        surface = _rough_surface(sensors[:, 0], c.surface.y0, modes)
        sdf = _sdf_to_curve(sensors[:, 0], sensors[:, 1], curve_x, curve_y, surface)
        expected = .5*(1+np.tanh(k*sdf))
        expected[np.isclose(sensors[:, 1], 0)] = 1
        expected[np.isclose(sensors[:, 1], .5)] = 0
        expected = expected.astype(np.float32).reshape(1, h, w)
        np.testing.assert_array_equal(expected, images[i])
        image_gap.append(float(np.max(abs(expected-images[i]))))
        with np.load(BASE/f'condition_{i:03d}/reference.npz') as f:
            np.testing.assert_array_equal(f['X'], xx)
            np.testing.assert_array_equal(f['Y'], yy)
            expected = np.stack([f['phi'][0], f['c'][0]], -1).reshape(-1, 2)
        gap = float(np.max(abs(expected-initial[i])))
        if gap > 3e-7:
            raise ValueError('Cached initial field/reference mismatch')
        initial_gap.append(gap)
        reference = json.loads((BASE/f'condition_{i:03d}/reference.json').read_text())
        reference_gate.append(reference['refinement_max_relative_l2'])
        if reference_gate[-1] >= c.experiment.reference_error_tolerance:
            raise ValueError('Unvalidated native reference')
    write('electro-spatial-input-audit.json', dict(status='passed', count=64,
        cached_images_independently_reconstructed=True, max_image_gap=max(image_gap),
        max_initial_reference_gap=max(initial_gap), max_native_refinement_error=max(reference_gate),
        new_reference_solves=0))
    print('INPUT_AUDIT_PASS', max(image_gap), max(initial_gap), max(reference_gate), flush=True)


def runs():
    records, samples = [], []
    first = ROOT/(PREFIX+'-'+MODES[0])
    initial_model = (first/'model_initial.eqx').read_bytes()
    initial_state = (first/'optimizer_initial.eqx').read_bytes()
    start = time.perf_counter()
    for mode in MODES:
        run = ROOT/(PREFIX+'-'+mode)
        summary = json.loads((run/'summary.json').read_text())
        c = json.loads((run/'config.json').read_text())
        if summary['iterations'] != 5000 or summary['training_uses_positive_time_labels']:
            raise ValueError('Incomplete or supervised training')
        assert c['sample']['spatial']['mode'] == mode
        assert (run/'model_initial.eqx').read_bytes() == initial_model
        assert (run/'optimizer_initial.eqx').read_bytes() == initial_state
        for name in json.loads((run/'source/files.json').read_text()):
            assert (run/'source'/name).read_bytes() == (REPO/name).read_bytes(), name
        with np.load(run/'sampling.npz') as f:
            sample = {k:f[k] for k in f.files}
        samples.append(sample)
        n, k, capacity = c['sample']['Nx']*c['sample']['Ny'], c['sample']['K_core'], c['sample']['K_pool']
        free = np.ones((c['sample']['Ny'], c['sample']['Nx']), dtype=bool)
        free[[0, -1]] = False
        free = free.ravel()
        expected_draws = (np.zeros(5000, dtype=int) if mode == 'full' else
            np.arange(5000) % capacity if mode == 'fixed_pool' else np.arange(5000))
        np.testing.assert_array_equal(sample['draw_ids'], expected_draws)
        seen = np.zeros(n, dtype=bool)
        for j, draw in enumerate(sample['entry_draw_ids']):
            ids = np.arange(n) if mode == 'full' else np.sort(np.random.RandomState(int(draw)).choice(n, k, replace=False))
            expected = ids[free[ids]]
            actual = sample['core_ids'][j][sample['core_mask'][j]]
            np.testing.assert_array_equal(actual, expected)
            seen[actual] = True
        assert int(seen.sum()) == summary['spatial_sampling']['unique_free_cores']
        records.append(dict(mode=mode, actual_draws=5000, distinct_core_sets=len(sample['entry_draw_ids']),
            exact_core_draw_replay=True, unique_free_cores=int(seen.sum()),
            coverage=float(seen.sum()/free.sum()), initial_model_and_optimizer_bytes_equal=True))
    for name in ('times', 'ic_ids', 'dirichlet_ids', 'side_ids'):
        for sample in samples[1:]:
            np.testing.assert_array_equal(sample[name], samples[0][name])
    for filename in ('online_coefficients.npy', 'online_accepted.npy'):
        a = np.load(first/filename)
        for mode in MODES[1:]:
            np.testing.assert_array_equal(np.load(ROOT/(PREFIX+'-'+mode)/filename), a)
    # The two minibatch controls share the entire initial32entry pool.
    for name in ('core_ids', 'core_mask', 'nonlocal_halo_counts'):
        np.testing.assert_array_equal(samples[1][name], samples[2][name][:32])
    write('electro-spatial-run-audit.json', dict(status='passed', records=records,
        initial_weights_and_adam_state_equal=True, all40000actual_conditions_equal=True,
        all5000time_batches_equal=True, IC_BC_coordinates_equal=True,
        fixed_rolling_first32entries_equal=True, seconds=time.perf_counter()-start))
    print('RUN_AUDIT_PASS', json.dumps(records), flush=True)


def cpu():
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    from ml_collections import ConfigDict
    from examples.electropolish2d.operator_training import _build_network, predict_fields
    from examples.electropolish2d.operator_data import load_dataset
    if jax.default_backend() != 'cpu':
        raise ValueError('Independent replay must use the CPU')
    checks = []
    for mode in MODES:
        run = ROOT/(PREFIX+'-'+mode)
        c = ConfigDict(json.loads((run/'config.json').read_text()))
        with jax.enable_x64(False):
            template = _build_network(c.arch, c.operator.image_shape, jax.random.key(0))
            net = eqx.tree_deserialise_leaves(run/'model_final.eqx', template)
        if not all(np.isfinite(np.asarray(a)).all() for a in jax.tree.leaves(eqx.filter(net, eqx.is_array))):
            raise FloatingPointError('Nonfinite checkpoint')
        data, metadata, _ = load_dataset(c)
        with np.load(run/'prediction.npz') as stored:
            saved = {k:stored[k] for k in stored.files}
        with jax.enable_x64(True):
            net = jax.tree.map(lambda a:a.astype(jnp.float64) if eqx.is_inexact_array(a) else a, net)
            x = jnp.asarray(data['points'].astype(np.float32).astype(np.float64))
            times = jnp.asarray(saved['time'].astype(np.float32).astype(np.float64))
            predict = eqx.filter_jit(lambda image, initial: predict_fields(net, image, x, times, initial, c))
            gaps = []
            for j, index in enumerate(metadata['eval_indices']):
                p = np.asarray(predict(jnp.asarray(data['images'][index], dtype=jnp.float64),
                    jnp.asarray(data['initial'][index], dtype=jnp.float64))).reshape(saved['prediction'][j].shape)
                gap = float(np.max(abs(p-saved['prediction'][j])))
                if gap > 2e-5:
                    raise ValueError(f'Independent CPU field gap{gap} exceeds declared2e-5')
                gaps.append(gap)
        checks.append(dict(mode=mode, cases=len(gaps), max_absolute_field_gap=max(gaps),
            replay='SavedFP32parameters promoted toFP64, identical rounded coordinates/times, independentCPU forward'))
    write('electro-spatial-cpu-replay.json', dict(status='passed', records=checks, tolerance=2e-5))
    print('CPU_REPLAY_PASS', json.dumps(checks), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('inputs', 'runs', 'cpu'))
    {'inputs':inputs, 'runs':runs, 'cpu':cpu}[parser.parse_args().action]()
