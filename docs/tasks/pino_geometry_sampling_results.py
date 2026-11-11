"""Evaluate and audit the matched channel and ceramic spatial controls."""
import argparse
import json
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    path = Path(path)
    partial = path.with_suffix('.partial')
    partial.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
    partial.replace(path)


def ceramic_inputs(area, wall, output):
    """Check both prepared banks before either arm consumes GPU time."""
    from examples.particle_corrosion2d.configs.train_physical_spaced_w256 import get_config as base
    from examples.particle_corrosion2d.configs.train_physical_spaced_w256_wall_cores import get_config as stratified
    from examples.particle_corrosion2d.train import load_bank
    from examples.particle_corrosion2d.online import pad_initial_bank
    cb, cw = base(), stratified()
    b, mb = load_bank(area, cb)
    w, mw = load_bank(wall, cw)
    if mb['runtime_seed'] != mw['runtime_seed'] or mb['geometries'] != mw['geometries']:
        raise ValueError('Ceramic initial geometry identities differ')
    for key in mb['fixed_keys']:
        np.testing.assert_array_equal(b[key], w[key])
    np.testing.assert_allclose(w['pde_weights'].sum(axis=-1, dtype=np.float64), 256, atol=2e-5, rtol=0)
    bp, wp = pad_initial_bank(b, cb), pad_initial_bank(w, cw)
    if {k: v.shape for k, v in bp.items()} != {k: v.shape for k, v in wp.items()}:
        raise ValueError('Ceramic fixed padding differs between arms')
    fraction = w['pde_weights'][:, :, 0]/2
    write(output, dict(status='passed', geometries=len(mb['geometries']), fixed_inputs_equal=True,
        padded_shapes_equal=True, near_wall_fraction_range=[float(fraction.min()), float(fraction.max())]))


def ceramic_eval(run, native):
    import equinox as eqx
    import jax
    from ml_collections import ConfigDict
    from nonlocal_pinn.training.runtime import configure_gpu_policy
    from examples.particle_corrosion2d.evaluate import load_references, make_evaluator
    from examples.particle_corrosion2d.geometry import boundary
    from examples.particle_corrosion2d.inputs import particle_wall_band
    from examples.particle_corrosion2d.network import build_network

    c = ConfigDict(read(run/'config.json'))
    r = read(run/'runtime.json')
    configure_gpu_policy(c.train.gpu_policy)
    jax.config.update('jax_enable_x64', False)
    jax.config.update('jax_default_matmul_precision', c.train.matmul_precision)
    model = eqx.tree_deserialise_leaves(run/'model_final.eqx', build_network(c, jax.random.PRNGKey(r['seed'])))
    report = {}
    for group, directory in (('local', Path(r['reference'])), ('native', native)):
        refs, manifest = load_references(directory, c)
        if group == 'native':
            predictions, scores = make_evaluator(c, refs)(model)
        else:
            scores = read(run/'summary.json')['final']
            predictions = []
            for i, ref in enumerate(refs):
                with np.load(run/f'prediction_{i:03d}.npz') as f:
                    np.testing.assert_array_equal(f['points'], ref['points'])
                    np.testing.assert_array_equal(f['times'], ref['times'])
                    predictions.append(f['prediction'])
        near_scores = []
        count = c.mesh.boundary_points
        theta = np.arange(count)*(2*np.pi/count)
        for pred, ref, row in zip(predictions, refs, manifest['cases'], strict=True):
            poly = np.stack([boundary(p, theta)[0] for p in row['geometry']['particles']])
            band = particle_wall_band(ref['points'], poly, c.nl.delta)
            v = ref['volumes'][band]
            p, y = pred[:, band].astype(float), ref['fields'][:, band].astype(float)
            t = ref['times']
            tw = np.r_[np.diff(t[:2])/2, (t[2:]-t[:-2])/2, np.diff(t[-2:])/2]
            def l2(a, b, w):
                return np.sqrt(np.sum(w[..., None]*(a-b)**2, axis=tuple(range(a.ndim-1)))
                               /np.sum(w[..., None]*b*b, axis=tuple(range(b.ndim-1))))
            error = np.r_[l2(p, y, tw[:, None]*v), l2(p[-1], y[-1], v)]
            near_scores.append(dict(zip(('phi_pooled','c_pooled','phi_terminal','c_terminal'), error.tolist(), strict=True)))
        report[group] = dict(overall=scores, near_wall_per_geometry=near_scores, near_wall_mean={
            k: float(np.mean([d[k] for d in near_scores])) for k in near_scores[0]})
    write(run/'spatial_reference_metrics.json', report)


def channel_audit(run):
    from ml_collections import ConfigDict
    from examples.channel_nsch2d.online import audit_training_inputs
    c = ConfigDict(read(run/'config.json'))
    write(run/'online-validation.json', audit_training_inputs(run, c))


def paired(paths, kind, output):
    """Verify cold states, fixed inputs, and physical geometry/time streams."""
    ref = paths[0]
    time_keys = ('indices', 'times', 'draw_ids', 'amplitudes') if kind == 'channel' else ('condition_ids','pool_ids','times')
    result = []
    for path in paths:
        for name in ('model_initial.eqx','optimizer_initial.eqx'):
            if (ref/name).read_bytes() != (path/name).read_bytes():
                raise ValueError(f'Cold initialization differs at {path}/{name}')
        with np.load(ref/'sampling.npz') as a, np.load(path/'sampling.npz') as b:
            for key in time_keys:
                np.testing.assert_array_equal(a[key], b[key])
        summary = read(path/'summary.json')
        if kind == 'channel':
            with np.load(path/'spatial_timing.npz') as f:
                update, prep = f['update'], f['batch_preparation']
            entry = dict(run=str(path), development=summary['metrics'],
                         reserved=read(path/'reserved'/'evaluation.json')['aggregate'],
                         spatial=summary['spatial'], peak_active_GiB=summary['device_memory_bytes']['peak_bytes_in_use']/1024**3,
                         warm_median_seconds=float(np.median((update+prep)[10:])))
        else:
            with np.load(ref/'online_sampling.npz') as a, np.load(path/'online_sampling.npz') as b:
                for key in ('geometry_draw_ids','spatial_versions'):
                    np.testing.assert_array_equal(a[key], b[key])
            # Check both the initial known data and the final rolled payload,
            # allowing only the graph targets and their closure to differ.
            from examples.particle_corrosion2d.online import FIXED_KEYS
            for name in ('online_initial_inputs.npz','online_final_inputs.npz'):
                with np.load(ref/name) as a, np.load(path/name) as b:
                    for key in FIXED_KEYS:
                        np.testing.assert_array_equal(a[key], b[key])
            timing = read(path/'timing.json')
            update, prep, online = (np.asarray(timing[k]) for k in
                ('update_seconds','sample_seconds','online_update_seconds'))
            entry = dict(run=str(path), errors=read(path/'spatial_reference_metrics.json'),
                         warm_median_seconds=float(np.median((update+prep+online)[100:])),
                         effective_mean_seconds=float(np.mean((update+prep+online)[100:])),
                         first_update_seconds=update[0], total_seconds=timing['total_seconds'],
                         peak_active_GiB=max(m['allocator']['peak_bytes_in_use'] for m in read(path/'memory.json'))/1024**3)
        result.append(entry)
    write(output, dict(status='passed', cold_state_geometry_and_time_draws_equal=True, arms=result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('ceramic-inputs','ceramic-eval','channel-audit'))
    parser.add_argument('path', nargs='+', type=Path)
    args = parser.parse_args()
    {'ceramic-inputs':ceramic_inputs, 'ceramic-eval':ceramic_eval, 'channel-audit':channel_audit}[args.action](*args.path)
