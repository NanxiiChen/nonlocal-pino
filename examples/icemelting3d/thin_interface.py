"""Matched three-dimensional sphere PINNs, sharing fresh core/time queries.

Both use the existing hard-IC tanh map and transformed residual. Nonlocal uses
the unchanged six-axis cubic kernel weights; local uses its analytic-chain-rule
AD Laplacian. The explicit homogeneous Neumann loss uses a one-sided flux in
the graph arm and the physical normal derivative in the AD arm. Its targets
are prescribed boundary data, never reference trajectories.
"""
import json
from pathlib import Path
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from examples.icemelting3d.losses import _compute_pde_latent
from examples.icemelting3d.phase import wrap_phase
from examples.icemelting3d.thin_interface_reference import graph_weight, parameters
from nonlocal_pinn.nn.arch import make_pinn
from nonlocal_pinn.training.optimizers import make_optimizer
from nonlocal_pinn.training.runtime import configure_gpu_policy


def validate(c):
    s = c.experiment
    if (c.nl.method not in ('sparse', 'local') or c.phase.formulation != 'ic_latent'
            or c.arch.type != 'mlp' or c.arch.use_hard_constraint
            or tuple(c.nondim.xc) != (.5, .5, .5)
            or not c.sample.Nx == c.sample.Ny == c.sample.Nz
            or c.sample.Nx % 2 or c.sample.Nx < 4
            or s.normalization != 'divide_by_N3'):
        raise ValueError('Thin-interface control requires centered scalar IC-latent MLP and even six-axis grid')
    counts = (c.train.iterations, c.sample.K_core, c.sample.Nt, s.boundary_points,
              s.evaluation_axis, s.evaluation_chunk, s.radial_points, s.radial_directions)
    if any(type(v) is not int or v < 1 for v in counts):
        raise ValueError('Positive integer study sampling/training counts required')
    values = (*parameters(c), c.physical.eps_star, s.boundary_weight)
    if not np.isfinite(values).all() or min(values) <= 0:
        raise ValueError('Positive finite physical parameters and boundary weight required')
    graph_weight(c)


def build_network(c):
    a = c.arch
    raw = make_pinn(spatial_dim=3, out_dim=1, trunk_type=a.type,
        hidden_dim=a.hidden_dim, num_layers=a.num_layers, activation=a.activation,
        use_spatial_emb=a.use_spatial_emb, use_temporal_emb=a.use_temporal_emb,
        spatial_emb_dim=a.spatial_emb_dim, spatial_scale=a.spatial_scale,
        temporal_emb_dim=a.temporal_emb_dim, temporal_scale=a.temporal_scale,
        tensor_split=a.tensor_split, key=jax.random.PRNGKey(c.train.seed))
    return wrap_phase(raw, c)


def sample_batch(c, step):
    # Identical host sampling in both modes; no halo points contribute PDE rows.
    rng = np.random.default_rng(np.random.SeedSequence([c.train.seed, 9317, step]))
    n, k = c.sample.Nx, c.sample.K_core
    ids = rng.integers(0, n, (k, 3), dtype=np.int32)
    times = (np.arange(c.sample.Nt)+rng.random(c.sample.Nt))/c.sample.Nt
    bc = rng.random((c.experiment.boundary_points, 3))
    face = np.arange(len(bc)) % 6
    axes, side = face//2, face % 2
    bc[np.arange(len(bc)), axes] = side
    normal = np.eye(3)[axes]*(2*side[:, None]-1)
    return ids, times.astype(np.float32), bc.astype(np.float32), normal.astype(np.float32)


def stencil(ids, c):
    n, k = c.sample.Nx, len(ids)
    offsets = np.concatenate((np.eye(3, dtype=np.int32), -np.eye(3, dtype=np.int32)))
    adjacent = ids[:, None, :]+offsets[None]
    valid = np.all((adjacent >= 0) & (adjacent < n), axis=-1)
    adjacent = np.where(valid[..., None], adjacent, ids[:, None, :])
    points = np.concatenate((ids, adjacent.reshape(-1, 3)))/(n-1)
    row = np.repeat(np.arange(k, dtype=np.int32), 6)
    col = np.arange(k, 7*k, dtype=np.int32)
    values = -graph_weight(c)*valid.ravel().astype(np.float32)
    diagonal = np.pad(-values.reshape(k, 6).sum(axis=1), (0, 6*k))
    return points.astype(np.float32), (row, col, values.astype(np.float32), diagonal.astype(np.float32))


def make_loss_fn(c):
    a, b, g = parameters(c)
    local, k, h = c.nl.method == 'local', c.sample.K_core, 1/(c.sample.Nx-1)

    def loss(net, points, sub, times, boundary, normal):
        residual, _ = _compute_pde_latent(net, points, times, sub, a, b, g)
        pde = jnp.mean(jnp.abs(residual[:k]/g))
        if local:
            def one(t):
                return jax.vmap(lambda x, v: jax.jvp(lambda y: net(y, t)[0], (x,), (v,))[1])(boundary, normal)
            flux = jax.vmap(one)(times)
        else:
            def one(t):
                field = lambda xx: jax.vmap(net, in_axes=(0, None))(xx, t)[:, 0]
                return (field(boundary)-field(boundary-h*normal))/h
            flux = jax.vmap(one)(times)
        bc = jnp.mean(flux**2)
        return pde+c.experiment.boundary_weight*bc, jnp.stack((pde, bc))
    return loss


def crossing(values, radius):
    mask = (values[..., :-1] >= 0) & (values[..., 1:] < 0)
    if not mask.any(axis=-1).all():
        raise ValueError('A radial evaluation ray has no outward zero crossing')
    index = mask.argmax(axis=-1)
    lo = np.take_along_axis(values, index[..., None], axis=-1)[..., 0]
    hi = np.take_along_axis(values, (index+1)[..., None], axis=-1)[..., 0]
    return radius[index]+(radius[index+1]-radius[index])*lo/(lo-hi)


def field_metrics(pred, truth):
    # Exclude t=0, which is prescribed exactly, from every aggregate curve.
    p, y = pred[1:], truth[1:]
    interface = np.abs(y) < .9
    relative = lambda a, b: float(np.linalg.norm(a-b)/np.linalg.norm(b))
    return dict(pooled=relative(p, y), terminal=relative(p[-1], y[-1]),
        interface=relative(p[interface], y[interface]),
        interface_terminal=relative(p[-1][interface[-1]], y[-1][interface[-1]]))


def train(c):
    validate(c)
    configure_gpu_policy(c.train.gpu_policy)
    jax.config.update('jax_default_matmul_precision', c.train.matmul_precision)
    refpath = Path(c.experiment.reference_path)
    saved = json.loads((refpath.parent/'config.json').read_text())
    current = json.loads(c.to_json_best_effort())
    if saved['physical'] != current['physical'] or saved['nondim'] != current['nondim']:
        raise ValueError('Thickness study reference physical/scaling mismatch')
    for name in ('Nx', 'Ny', 'Nz'):
        if saved['sample'][name] != c.sample[name]:
            raise ValueError(f'Thickness study reference graph {name} mismatch')
    for name in ('delta', 'kernel'):
        if saved['nl'][name] != c.nl[name]:
            raise ValueError(f'Thickness study reference graph {name} mismatch')
    with np.load(refpath, allow_pickle=False) as f:
        ref = {key: f[key] for key in f.files}
    run = Path(c.log.log_dir)/c.log.run_name
    run.mkdir(parents=True, exist_ok=False)
    (run/'config.json').write_text(c.to_json_best_effort()+'\n')
    net = build_network(c)
    opt = make_optimizer(c)
    state = opt.init(eqx.filter(net, eqx.is_array))
    eqx.tree_serialise_leaves(run/'model_initial.eqx', net)
    eqx.tree_serialise_leaves(run/'optimizer_initial.eqx', state)
    loss_fn = make_loss_fn(c)

    @eqx.filter_jit
    def step(net, state, points, sub, times, boundary, normal):
        (value, parts), grad = eqx.filter_value_and_grad(loss_fn, has_aux=True)(
            net, points, sub, times, boundary, normal)
        updates, state = opt.update(grad, state, net)
        return eqx.apply_updates(net, updates), state, value, parts

    @eqx.filter_jit
    def predict(net, points):
        return jax.vmap(lambda t: jax.vmap(net, in_axes=(0, None))(points, t)[:, 0])(
            jnp.asarray(ref['time'], dtype=jnp.float32))

    def prediction(points):
        chunks, count = [], c.experiment.evaluation_chunk
        for start in range(0, len(points), count):
            xx = points[start:start+count]
            actual = len(xx)
            xx = np.pad(xx, ((0, count-actual), (0, 0)), mode='edge')
            chunks.append(np.asarray(predict(net, jnp.asarray(xx, dtype=jnp.float32)))[:, :actual])
        return np.concatenate(chunks, axis=1).astype(np.float64)

    timings, prep_times, records = [], [], []
    trace = dict(ids=[], times=[], boundary=[], normal=[])
    started = time.perf_counter()
    with (run/'loss.jsonl').open('x') as stream:
        for i in range(c.train.iterations):
            begin = time.perf_counter()
            ids, times, boundary, normal = sample_batch(c, i)
            for name, array in zip(trace, (ids, times, boundary, normal), strict=True):
                trace[name].append(array)
            if c.nl.method == 'local':
                points, sub = ids.astype(np.float32)/(c.sample.Nx-1), None
            else:
                points, sub = stencil(ids, c)
            payload = jax.device_put((points, sub, times, boundary, normal))
            jax.block_until_ready(payload)
            prep_times.append(time.perf_counter()-begin)
            begin = time.perf_counter()
            net, state, value, parts = step(net, state, *payload)
            jax.block_until_ready((net, state, value, parts))
            timings.append(time.perf_counter()-begin)
            if not np.isfinite(float(value)):
                raise FloatingPointError(f'Nonfinite study loss at update{i+1}')
            if (i+1) % 100 == 0:
                rec = dict(step=i+1, loss=float(value), parts=np.asarray(parts).tolist())
                stream.write(json.dumps(rec, allow_nan=False)+'\n'); stream.flush()
                print('TRAIN', json.dumps(rec), flush=True)
            if (i+1) % c.log.eval_every == 0 or i+1 == c.train.iterations:
                pred = prediction(ref['points'])
                record = dict(step=i+1, **field_metrics(pred, ref['local']))
                records.append(record)
                with (run/'metrics.jsonl').open('a') as output:
                    output.write(json.dumps(record, allow_nan=False)+'\n')
                eqx.tree_serialise_leaves(run/f'model_{i+1:06d}.eqx', net)
                print('EVAL', json.dumps(record), flush=True)
    rays = prediction(ref['rays'].reshape(-1, 3)).reshape(ref['local_rays'].shape)
    metrics = field_metrics(pred, ref['local'])
    eqx.tree_serialise_leaves(run/'model_final.eqx', net)
    np.savez_compressed(run/'prediction.npz', phi=pred, rays=rays)
    np.savez_compressed(run/'timing.npz', update=timings, preparation=prep_times)
    np.savez_compressed(run/'sampling.npz', **{name: np.stack(values) for name, values in trace.items()})
    target_radius = crossing(ref['local_rays'], ref['radius'])
    counts = np.sum((rays[..., :-1] >= 0) & (rays[..., 1:] < 0), axis=-1)[1:]
    crossing_info = dict(missing=int(np.sum(counts == 0)), multiple=int(np.sum(counts > 1)),
                         total=int(counts.size))
    metrics['radius_relative_rmse'] = None
    if not crossing_info['missing']:
        predicted_radius = crossing(rays, ref['radius'])
        metrics['radius_relative_rmse'] = float(np.sqrt(np.mean(
            ((predicted_radius[1:]-target_radius[1:])/target_radius[1:])**2)))
    summary = dict(method=c.nl.method, epsilon=c.physical.eps_star,
        width_phi90=2*np.arctanh(.9)*np.sqrt(2)*c.physical.eps_star,
        seed=c.train.seed, steps=c.train.iterations, metrics=metrics,
        native_comparison=field_metrics(pred, ref['native']),
        native_baseline=field_metrics(ref['native'], ref['local']),
        crossing_info=crossing_info,
        median_step_seconds=float(np.median(np.asarray(timings[20:])+np.asarray(prep_times[20:]))),
        first_step_seconds=timings[0], elapsed_seconds=time.perf_counter()-started,
        gpu=str(jax.devices()[0].device_kind), memory=jax.devices()[0].memory_stats(),
        physical_boundary='homogeneous Neumann; graph secant or local AD normal derivative',
        cold_start=True, final_model='model_final.eqx')
    (run/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    print('COMPLETE', json.dumps(summary), flush=True)
