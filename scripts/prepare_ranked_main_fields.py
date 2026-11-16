"""Extend original main-figure weights to existing test banks on an explicit backend.

No training or reference integration. Replays an original saved prediction before
extending inference. All paths and all phase scores are saved for reproduction.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np

REPO = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def error(prediction, reference):
    p, r = np.asarray(prediction, dtype=float), np.asarray(reference, dtype=float)
    if p.shape != r.shape or not np.isfinite(p).all() or not np.isfinite(r).all():
        raise ValueError('Phase arrays must be finite and identically shaped')
    denominator = float(np.sum(r*r))
    if denominator <= 0:
        raise ValueError('Zero reference norm')
    return float(np.sqrt(np.sum((p-r)**2)/denominator))


def select(records):
    order = sorted(records, key=lambda r: (r['error'], r['index']))
    if len(order) < 3 or len({r['index'] for r in order}) != len(order):
        raise ValueError('Need at least three distinct samples')
    selected = [order[0], order[(len(order)-1)//2], order[-1]]
    return [dict(record, rank_label=label) for record, label in
            zip(selected, ('Best', 'Middle', 'Worst'), strict=True)]


def ceramic(output):
    import equinox as eqx
    import jax
    from ml_collections import ConfigDict
    from examples.particle_corrosion2d.evaluate import load_references, make_evaluator
    from examples.particle_corrosion2d.network import build_network

    spec = read(REPO/'artifacts/particle_corrosion_main_figure.json')
    c = ConfigDict(read(spec['config']))
    jax.config.update('jax_enable_x64', c.train.compute_dtype == 'float64')
    jax.config.update('jax_default_matmul_precision', c.train.matmul_precision)
    net = eqx.tree_deserialise_leaves(spec['checkpoint'], build_network(c, jax.random.key(0)))
    original_refs, _ = load_references(Path(spec['reference_manifest']).parent, c)
    replay, _ = make_evaluator(c, original_refs[:1])(net)
    with np.load(spec['records'][0]['prediction']) as a:
        gap = float(np.max(np.abs(replay[0]-a['prediction'])))
    if gap > 2e-5:
        raise ValueError(f'Original Ceramic checkpoint replay failed: {gap}')
    root = Path('/root/autodl-tmp/data/particle_corrosion2d/local_test64_20261111-01')
    refs, manifest = load_references(root, c)
    if len(refs) != 64 or manifest['backend'] != 'local_p1_lumped_fem':
        raise ValueError('Require established 64 local-FEM cases')
    records = []
    predictions, _ = make_evaluator(c, refs)(net)
    for i, (ref, entry, pred) in enumerate(zip(refs, manifest['cases'], predictions, strict=True)):
        path = output/f'ceramic_prediction_{i:03d}.npz'
        np.savez_compressed(path, prediction=pred, points=ref['points'], times=ref['times'])
        record = dict(index=i, prediction=str(path), reference=str(root/entry['file']),
                      error=error(pred[..., 0], ref['fields'][..., 0]))
        records.append(record)
        print(json.dumps(dict(case='ceramic', **record)), flush=True)
    write(output/'ceramic_ranking.json', dict(checkpoint=spec['checkpoint'], config=spec['config'],
        reference_manifest=str(root/'manifest.json'), count=64, replay_gap=gap,
        backend=jax.default_backend(), records=records, selected=select(records)))


def melting(output):
    import equinox as eqx
    import jax
    from ml_collections import ConfigDict
    from examples.icemelting3d.operator_data import build_network
    from examples.icemelting3d.operator_training import make_predictor

    spec = read(REPO/'artifacts/melting_main_figure.json')
    c = ConfigDict(read(spec['config']))
    jax.config.update('jax_enable_x64', False)
    jax.config.update('jax_default_matmul_precision', c.train.matmul_precision)
    net = eqx.tree_deserialise_leaves(spec['checkpoint'], build_network(c, jax.random.key(0)))
    predict = make_predictor(c)
    backend = jax.default_backend()
    root = Path('/root/autodl-tmp/data/icemelting3d/operator_ellipsoids_test64_20261114_v1')
    manifest = read(root/'manifest.json')
    original = read(spec['reference_manifest'])
    for key in ('physical', 'nondim', 'nl', 'grid', 'initial', 'image_shape', 'snapshots', 'evaluation_axis_points'):
        if manifest['identity'][key] != original['identity'][key]:
            raise ValueError('Reference identity mismatch: '+key)
    # CPU and the historical GPU execution do not reproduce pointwise FP32
    # values to the original same-GPU tolerance. Keep a separate, explicit
    # backend audit; do not silently label these new fields as cached GPU data.
    import importlib.util
    frozen_path = Path(spec['checkpoint']).parent/'source/examples/icemelting3d/operator_training.py'
    module_spec = importlib.util.spec_from_file_location('ranked_frozen_melting', frozen_path)
    frozen = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(frozen)
    original_predict = frozen.make_predictor(c)
    backend_audit = []
    for path in sorted(Path(spec['checkpoint']).parent.glob('prediction_*.npz')):
        with np.load(path) as a:
            replay = predict(net, a['parameters'], a['points'], a['time'])
            old_code = original_predict(net, a['parameters'], a['points'], a['time'])
            np.testing.assert_array_equal(replay, old_code)
            gap = float(np.max(np.abs(replay-a['prediction'])))
            relative_gap = error(replay, a['prediction'])
            score_gap = abs(error(replay, a['target'])-error(a['prediction'], a['target']))
            if backend == 'gpu' and gap > spec['replay_tolerance']:
                raise ValueError(f'GPU replay fails original tolerance: {path}, {gap}')
            # A declared absolute relative-L2 change of 0.001 is the maximum
            # accepted CPU preview drift, not a claim of bitwise GPU replay.
            if relative_gap > .001 or score_gap > .001:
                raise ValueError(f'CPU preview differs materially: {path}, {relative_gap}, {score_gap}')
            backend_audit.append(dict(source=str(path), maximum_phase_difference=gap,
                relative_field_difference=relative_gap, absolute_score_difference=score_gap,
                frozen_and_current_arrays_identical=True))
    audit_path = output/'melting_replay_audit.json'
    write(audit_path, dict(backend=backend, original_same_gpu_tolerance=spec['replay_tolerance'],
        original_pointwise_tolerance_passed=all(r['maximum_phase_difference'] <= spec['replay_tolerance'] for r in backend_audit),
        relative_cpu_preview_tolerance=.001,
        note='CPU diagnostic is not selected for final preview; require original tolerance for GPU.',
        frozen_predictor=str(frozen_path), records=backend_audit))
    records = []
    if len(manifest['test_indices']) != 64:
        raise ValueError('Require complete 64-case test cohort')
    for i in manifest['test_indices']:
        source = root/f'condition_{i:03d}.npz'
        with np.load(source) as a:
            np.testing.assert_array_equal(a['parameters'], manifest['parameters'][i])
            pred = predict(net, a['parameters'], a['points'], a['time'])
            path = output/f'melting_prediction_{i:03d}.npz'
            np.savez_compressed(path, prediction=pred, points=a['points'], time=a['time'], parameters=a['parameters'])
            record = dict(index=i, prediction=str(path), reference=str(source), error=error(pred, a['phi']))
        records.append(record)
        print(json.dumps(dict(case='melting', **record)), flush=True)
    selected = select(records)
    write(output/'melting_ranking.json', dict(checkpoint=spec['checkpoint'], config=spec['config'],
        reference_manifest=str(root/'manifest.json'), count=64, replay_audit=str(audit_path),
        backend=backend, records=records, selected=selected))
    axis = np.linspace(0, 1, spec['surface_axis_points'], dtype=np.float32)
    points = np.stack(np.meshgrid(axis, axis, axis, indexing='ij'), -1).reshape(-1, 3)
    for record in selected:
        i = record['index']
        with np.load(record['reference']) as a:
            dense = predict(net, a['parameters'], points, a['time'])
            sliced = predict(net, a['parameters'], a['slice_points'], a['time'])
            plane = dense.reshape(5, len(axis), len(axis), len(axis))[:, :, :, spec['slice_index']].reshape(5, -1)
            plane_gap = float(np.max(np.abs(plane-sliced)))
            if plane_gap > spec['replay_tolerance']:
                raise ValueError(f'Dense/slice consistency failed: {plane_gap}')
            np.savez_compressed(output/f'melting_dense_{i:03d}.npz', phi=dense.reshape(5, len(axis), len(axis), len(axis)),
                                slice_prediction=sliced, axis=axis, time=a['time'])
        print(json.dumps(dict(case='melting_dense', index=i, plane_gap=plane_gap)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--case', choices=('ceramic', 'melting'), required=True)
    parser.add_argument('--backend', choices=('cpu', 'gpu'), required=True)
    args = parser.parse_args()
    import jax
    if jax.default_backend() != args.backend:
        raise RuntimeError('The requested and active inference backend must agree')
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    {'ceramic': ceramic, 'melting': melting}[args.case](args.output)
    print(json.dumps(dict(completed=args.case, seconds=time.monotonic()-started)), flush=True)
