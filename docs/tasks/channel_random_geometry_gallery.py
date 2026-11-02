"""CPU-only snapshot predictions on explicitly selected reserved geometries."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main(run, step, indices, grid, requested_times, output):
    os.environ['JAX_PLATFORMS'] = 'cpu'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    available = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0, available[-4:])
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import numpy as np
    from ml_collections import ConfigDict
    from scipy.interpolate import RegularGridInterpolator
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from examples.channel_nsch2d.cohort import load
    from examples.channel_nsch2d.data import geometry_data
    from examples.channel_nsch2d.network import build_network, encode, fields
    from examples.channel_nsch2d.problem import walls, initial_field, flow_rate, identity

    run, output = Path(run), Path(output)
    checkpoint = run/f'ckpt_{step:06d}.eqx'
    c = ConfigDict(json.loads((run/'config.json').read_text()))
    cohort = load(c.operator.validation_manifest, c)
    if (not indices or len(set(indices)) != len(indices)
            or any(i < 0 or i >= cohort['count'] for i in indices)):
        raise ValueError('Unique reserved manifest indices required')
    nx, ny = grid
    times = np.asarray(requested_times, dtype=np.float64)
    if (min(grid) < 8 or times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all()
            or times[0] != 0 or np.any(np.diff(times) <= 0) or times[-1] > c.physical.terminal_time):
        raise ValueError('Explicit mapped grid and increasing physical times starting at zero required')
    if output.exists():
        raise FileExistsError(output)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    replay = json.loads((run/f'partial_{step:06d}.json').read_text())
    if replay['status'] != 'partial_checkpoint_metric_replay_passed':
        raise ValueError('Nominal logged-metric replay must pass before snapshot illustration')
    jax.config.update('jax_enable_x64', c.train.compute_dtype == 'float64')
    jax.config.update('jax_default_matmul_precision', c.train.matmul_precision)
    if jax.default_backend() != 'cpu':
        raise RuntimeError('Gallery must leave GPU training undisturbed')
    dtype = np.dtype(c.train.compute_dtype)
    key, _ = jax.random.split(jax.random.PRNGKey(c.train.seed))
    net = eqx.tree_deserialise_leaves(checkpoint, build_network(c, key))
    nodes = jnp.asarray(times, dtype=dtype)
    predict = eqx.filter_jit(lambda model, d, q: fields(model, encode(model, d), q,
                            nodes, d['amplitudes'], c))
    xx, ee = np.meshgrid((np.arange(nx)+.5)*c.geometry.length/nx,
                         (np.arange(ny)+.5)/ny, indexing='ij')
    section = np.linspace(0, c.geometry.length, 5)
    roots, weights = np.polynomial.legendre.leggauss(64)
    sx, se = np.meshgrid(section, (roots+1)/2, indexing='ij')
    output.mkdir(parents=True)
    records, samples = [], []
    for index in indices:
        case = cohort['cases'][index]
        amplitudes = np.asarray(case['amplitudes'], dtype=np.float64)
        raw, _ = geometry_data(amplitudes, c)
        data = {k: jnp.asarray(v, dtype=v.dtype if np.issubdtype(v.dtype, np.integer) else dtype)
                for k, v in raw.items()}
        bottom, top = walls(xx, amplitudes, c, xp=np)
        points = np.stack((xx, bottom+ee*(top-bottom)), axis=-1).reshape(-1, 2)
        sb, st = walls(sx, amplitudes, c, xp=np)
        section_points = np.stack((sx, sb+se*(st-sb)), axis=-1).reshape(-1, 2)
        query = np.concatenate((points, raw['boundary'][:, :2], section_points))
        values = np.asarray(predict(net, data, jnp.asarray(query, dtype=dtype)), dtype=np.float64)
        if values.shape != (len(times), len(query), 5) or not np.isfinite(values).all():
            raise ValueError('Finite aligned five-field prediction required')
        n, nb = len(points), len(raw['boundary'])
        prediction, boundary = values[:, :n], values[:, n:n+nb]
        velocity = values[:, n+nb:, 2].reshape(len(times), len(section), 64)
        flux = np.sum(velocity*(st-sb)[None]*weights[None, None]/2, axis=-1)
        flux_target = np.asarray(flow_rate(times, c, xp=np))[:, None]
        wall = raw['boundary'][:, 4] == 0
        inlet = raw['boundary'][:, 4] == 1
        y = raw['boundary'][inlet, 1]
        inlet_velocity = 6*flow_rate(times, c, xp=np)[:, None]*y*(c.geometry.height-y)/c.geometry.height**3
        gaps = dict(initial_phase=float(np.max(abs(prediction[0, :, 0]-initial_field(points.astype(dtype), c, xp=np)))),
            initial_velocity=float(np.max(abs(prediction[0, :, 2:4]))),
            wall_velocity=float(np.max(abs(boundary[:, wall, 2:4]))),
            inlet_phi=float(np.max(abs(boundary[:, inlet, 0]-1))),
            inlet_mu=float(np.max(abs(boundary[:, inlet, 1]))),
            inlet_velocity=float(np.max(abs(boundary[:, inlet, 2]-inlet_velocity))),
            section_flux=float(np.max(abs(flux-flux_target))))
        if max(gaps.values()) > 2e-5:
            raise ValueError(f'Reserved illustration hard-constraint check failed: {index}/{gaps}')
        physical = ConfigDict(c.to_dict())
        physical.geometry.upper, physical.geometry.lower = map(float, amplitudes)
        path = output/f'case_{index:03d}.npz'
        np.savez_compressed(path, fields=prediction, points=points, amplitudes=amplitudes,
            time=times, identity_json=json.dumps(identity(physical)))
        records.append(dict(index=index, amplitudes=amplitudes.tolist(), payload=str(path),
            hard_constraint_max_gaps=gaps, phase_range=[float(prediction[..., 0].min()), float(prediction[..., 0].max())]))
        samples.append((amplitudes, prediction))
        print(json.dumps(records[-1]), flush=True)

    # Display interpolation only; all diagnostics above use raw physical queries.
    dx, dy = max(256, nx), max(64, ny)
    mx, me = np.meshgrid(np.linspace(0, c.geometry.length, dx+1), np.linspace(0, 1, dy+1), indexing='ij')
    cx, ce = np.meshgrid((np.arange(dx)+.5)*c.geometry.length/dx, (np.arange(dy)+.5)/dy, indexing='ij')
    xq, eq = xx[:, 0], ee[0]
    display = np.stack((np.clip(cx, xq[0], xq[-1]), np.clip(ce, eq[0], eq[-1])), axis=-1)
    for label in ('phi', 'speed'):
        fig, axes = plt.subplots(len(indices), len(times), figsize=(3.2*len(times), 1.75*len(indices)),
                                 squeeze=False, layout='constrained')
        high = 1. if label == 'phi' else max(float(np.linalg.norm(v[..., 2:4], axis=-1).max()) for _, v in samples)
        for row, (amplitudes, values) in enumerate(samples):
            bottom, top = walls(mx, amplitudes, c, xp=np)
            yy = bottom+me*(top-bottom)
            for col, t in enumerate(times):
                field = values[col, :, 0] if label == 'phi' else np.linalg.norm(values[col, :, 2:4], axis=-1)
                smooth = RegularGridInterpolator((xq, eq), field.reshape(nx, ny))(display)
                ax = axes[row, col]
                mesh = ax.pcolormesh(mx.T, yy.T, smooth.T, shading='flat',
                    cmap='coolwarm' if label == 'phi' else 'viridis', vmin=-1 if label == 'phi' else 0, vmax=high, rasterized=True)
                wx = np.linspace(0, c.geometry.length, 401)
                wb, wt = walls(wx, amplitudes, c, xp=np)
                ax.plot(wx, wb, color='.2', lw=.8); ax.plot(wx, wt, color='.2', lw=.8)
                ax.set(xlim=(0, c.geometry.length), ylim=(0, c.geometry.height), aspect='equal', xticks=[], yticks=[])
                if row == 0: ax.set_title(f't={t:g}')
                if col == 0: ax.set_ylabel(f'Case {indices[row]}\nupper={amplitudes[0]:.3f}\nlower={amplitudes[1]:.3f}')
        fig.colorbar(mesh, ax=axes.ravel().tolist(), shrink=.85, pad=.012)
        fig.suptitle(f'PINO {label} predictions: checkpoint {step:,}; fixed reserved random geometries\nPrediction-only illustration; no per-geometry FEM error available')
        fig.savefig(output/f'random-{label}.png', dpi=200)
        fig.savefig(output/f'random-{label}.pdf'); plt.close(fig)
    report = dict(status='reserved_geometry_prediction_gallery_completed', run=str(run), checkpoint=str(checkpoint),
        backend=jax.default_backend(), indices=indices, query_grid=list(grid), time=times.tolist(), cases=records,
        selection='Explicit reserved indices fixed before reading predictions',
        scope='Illustrations only; no FEM solve/reference, optimizer recovery, population accuracy or training completion claim')
    (output/'gallery.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    shutil.copyfile(Path(__file__), output/Path(__file__).name)
    print(json.dumps(dict(status=report['status'], output=str(output))), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True); parser.add_argument('--step', required=True, type=int)
    parser.add_argument('--indices', required=True, nargs='+', type=int)
    parser.add_argument('--query-grid', required=True, nargs=2, type=int)
    parser.add_argument('--times', required=True, nargs='+', type=float); parser.add_argument('--output', required=True)
    args = parser.parse_args(); main(args.run, args.step, args.indices, args.query_grid, args.times, args.output)
