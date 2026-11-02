"""Frozen full-time PDE/BC audit and two CH-only causal reintegrations on CPU."""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from scipy import sparse
from scipy.interpolate import RegularGridInterpolator
from ml_collections import ConfigDict
from examples.channel_nsch2d.data import geometry_data
from examples.channel_nsch2d.network import build_network, encode, fields, spatial_jets, decode, lift, spatial_field_jets
from examples.channel_nsch2d.problem import boundary, material, walls, flow_rate, validate_config
from examples.channel_nsch2d.losses import make_loss_fn
from docs.tasks.channel_frozen_ch import FrozenCH, difference_matrix


def rms(values, weight):
    if weight.sum() <= 0:
        raise ValueError('Positive diagnostic measure required')
    return np.sqrt(np.sum(values**2*weight, axis=-1)/np.sum(weight))


def boundary_matrix(indices, weights, n):
    weights = np.asarray(weights, dtype=np.float64)
    rows = np.repeat(np.arange(len(indices)), indices.shape[1])
    return sparse.coo_matrix((weights.ravel(), (rows, indices.ravel())), shape=(len(indices), n)).tocsr()


def freeze_predict(net, data, points, times, c):
    encoded = eqx.filter_jit(encode)(net, data)
    x = jnp.asarray(points, dtype=c.train.compute_dtype)
    h, hx, hy = eqx.filter_jit(spatial_jets)(net, encoded, x)

    @eqx.filter_jit
    def cached(n, h, hx, hy, x, t, a):
        fn = lambda prepared, point: lift(decode(n, prepared, t, c), point, t, a, c)
        ex = jnp.broadcast_to(jnp.array([1., 0.], dtype=x.dtype), x.shape)
        ey = jnp.broadcast_to(jnp.array([0., 1.], dtype=x.dtype), x.shape)
        value, dx = jax.jvp(fn, (h, x), (hx, ex))
        _, dy = jax.jvp(fn, (h, x), (hy, ey))
        physical = jnp.stack((value[..., 0], value[..., 1], dy[..., 2], -dx[..., 2], value[..., 3]), axis=-1)
        return jnp.concatenate((physical, dx[..., :1], dy[..., :1], dx[..., 1:2], dy[..., 1:2]), axis=-1)

    result = np.empty((len(times), len(points), 9), dtype=np.float32)
    for start in range(0, len(times), 4):
        actual = min(4, len(times) - start)
        batch = np.pad(times[start:start + actual], (0, 4 - actual), mode='edge')
        result[start:start + actual] = np.asarray(cached(net, h, hx, hy, x,
            jnp.asarray(batch, dtype=c.train.compute_dtype), data['amplitudes']))[:actual]
    ids = np.linspace(0, len(points) - 1, 64, dtype=int)
    ti = np.array([20, 120, 240])
    direct = np.asarray(eqx.filter_jit(lambda n, e, x, t, a: fields(n, e, x, t, a, c))(
        net, encoded, x[ids], jnp.asarray(times[ti], dtype=c.train.compute_dtype), data['amplitudes']))
    gap = float(np.max(np.abs(direct - result[ti][:, ids, :5])))
    if gap > 5e-6:
        raise ValueError(f'Frozen feature chain rule versus canonical fields failed: {gap}')
    return result, encoded, gap


def boundary_maps(values, target, b, gx, gy, idx):
    n = values.shape[1]
    Bx, By = boundary_matrix(idx, gx, n), boundary_matrix(idx, gy, n)
    dx = np.stack([np.asarray(Bx@v) for v in values]) - np.asarray(Bx.sum(axis=1)).ravel()[None, :, None]*target[..., :5]
    dy = np.stack([np.asarray(By@v) for v in values]) - np.asarray(By.sum(axis=1)).ravel()[None, :, None]*target[..., :5]
    normal = dx*b[None, :, 2, None] + dy*b[None, :, 3, None]
    return normal, dx, dy


def audit_case(net, raw, c, times, output, label, *, reintegrate):
    data = {k: jnp.asarray(v, dtype=v.dtype if np.issubdtype(v.dtype, np.integer)
                          else c.train.compute_dtype) for k, v in raw.items()}
    coarse_c = ConfigDict(c.to_dict()); coarse_c.sample.Nx, coarse_c.sample.Ny = 96, 40
    coarse, _ = geometry_data(np.asarray(raw['amplitudes'], dtype=np.float64), coarse_c)
    dense_c = ConfigDict(c.to_dict()); dense_c.sample.boundary_points = 192
    dense_b, dense_w = boundary(np.asarray(raw['amplitudes']), dense_c)
    from examples.channel_nsch2d.operators import first_derivative_stencils
    dense_i, dense_x, dense_y, _ = first_derivative_stencils(raw['points'], dense_b[:, :2], c)
    groups = [raw['points'], coarse['points'], raw['boundary'][:, :2], dense_b[:, :2]]
    ends = np.cumsum([len(p) for p in groups]); starts = np.r_[0, ends[:-1]]
    selections = [slice(a, b) for a, b in zip(starts, ends, strict=True)]
    print(json.dumps({'stage': 'frozen_prediction', 'case': label, 'query_nodes': int(ends[-1])}), flush=True)
    all_fields, encoded, primal_gap = freeze_predict(net, data, np.concatenate(groups), times, c)
    physical = all_fields[:, selections[0], :5]
    coarse_fields = all_fields[:, selections[1]]
    trained_boundary = all_fields[:, selections[2]]
    dense_fields = all_fields[:, selections[3]]
    print(json.dumps({'stage': 'full_time_residuals', 'case': label, 'primal_gap': primal_gap}), flush=True)
    n = len(raw['points'])
    D = [difference_matrix(raw['neighbors'], raw[name].astype(np.float64), n)
         for name in ('gradient_x', 'gradient_y', 'laplacian')]
    weight = np.asarray(raw['transport_measure'], dtype=np.float64)*raw['interior']
    btm, top = walls(raw['points'][:, 0], raw['amplitudes'], c, xp=np)
    eta = (raw['points'][:, 1] - btm)/(top - btm)
    near = (raw['points'][:, 0] < .2) | (raw['points'][:, 0] > c.geometry.length - .2) | (eta < .1) | (eta > .9)
    scales = np.array([c.loss.phase_scale, c.loss.chemical_scale, c.loss.momentum_scale,
                       c.loss.momentum_scale, c.loss.continuity_scale])
    A, kappa = material(c)
    rows, maps = [], []
    for index in range(1, len(times)):
        value = physical[index].astype(np.float64)
        dx, dy, lap = [matrix@value for matrix in D]
        dt = ((value - physical[0]) if index == 1 else
              (1.5*value - 2*physical[index - 1] + .5*physical[index - 2]))/c.time_discretization.dt
        phi, mu, u, v = value[:, :4].T
        transport = D[0]@(phi*u) + D[1]@(phi*v)
        residual = np.column_stack((dt[:, 0] + transport - c.physical.mobility*lap[:, 1],
            mu - A*(phi**3 - phi) + kappa*lap[:, 0],
            c.physical.density*(dt[:, 2] + u*dx[:, 2] + v*dy[:, 2]) + dx[:, 4] - c.physical.viscosity*lap[:, 2] + phi*dx[:, 1],
            c.physical.density*(dt[:, 3] + u*dx[:, 3] + v*dy[:, 3]) + dy[:, 4] - c.physical.viscosity*lap[:, 3] + phi*dy[:, 1],
            dx[:, 2] + dy[:, 3]))
        regions = {}
        for name, mask in [('all_interior', np.ones(n, dtype=bool)), ('predicted_interface', np.abs(phi) < .9),
                           ('predicted_bulk', np.abs(phi) > .99), ('near_boundary', near), ('away_from_boundary', ~near)]:
            w = weight*mask
            regions[name] = dict(raw_rms=rms(residual.T, w).tolist(),
                scaled_mse=(rms((residual/scales).T, w)**2).tolist(),
                scaled_residual_energy_share=float(np.sum((residual[:, :4]/scales[:4])**2*w[:, None])/
                    np.sum((residual[:, :4]/scales[:4])**2*weight[:, None])))
        rows.append(dict(time=float(times[index]), regions=regions,
                         phase_term_rms=rms(np.stack((dt[:, 0], transport, c.physical.mobility*lap[:, 1])), weight).tolist()))
        if index in (20, 60, 120, 180, 240):
            maps.append(residual.astype(np.float32))
    normal, bx, by = boundary_maps(physical[1:].astype(np.float64), trained_boundary[1:].astype(np.float64),
        raw['boundary'], raw['boundary_gradient_x'], raw['boundary_gradient_y'], raw['boundary_neighbors'])
    dn, dbx, dby = boundary_maps(physical[1:].astype(np.float64), dense_fields[1:].astype(np.float64),
        dense_b, dense_x, dense_y, dense_i)

    def bstats(target, b, w, normal, dx, dy):
        wall, out = b[:, 4] == 0, b[:, 4] == 2
        traction = (-target[..., 4] + 2*c.physical.viscosity*dx[..., 2])**2 + (c.physical.viscosity*(dy[..., 2] + dx[..., 3]))**2
        series = [rms(normal[..., 0], w*wall)**2, rms(normal[..., 1], w*wall)**2,
                  np.sum(traction*w*out, axis=1)/np.sum(w*out), rms(normal[..., 0], w*out)**2,
                  rms(normal[..., 1], w*out)**2]
        scale = np.array([c.loss.boundary_phi_scale, c.loss.boundary_mu_scale, c.loss.traction_scale,
                          c.loss.boundary_phi_scale, c.loss.boundary_mu_scale])
        return dict(raw_rms=np.sqrt(np.mean(series, axis=1)).tolist(),
                    scaled_mse=(np.mean(series, axis=1)/scale**2).tolist(),
                    scaled_per_time=(np.asarray(series).T/scale**2).tolist())

    trained = bstats(trained_boundary[1:], raw['boundary'], raw['boundary_weights'], normal, bx, by)
    dense = bstats(dense_fields[1:], dense_b, dense_w, dn, dbx, dby)
    physical_normal_phi = dense_fields[1:, :, 5]*dense_b[None, :, 2] + dense_fields[1:, :, 6]*dense_b[None, :, 3]
    physical_normal_mu = dense_fields[1:, :, 7]*dense_b[None, :, 2] + dense_fields[1:, :, 8]*dense_b[None, :, 3]
    continuous_boundary = {}
    for name, mask in [('wall', dense_b[:, 4] == 0), ('outlet', dense_b[:, 4] == 2)]:
        continuous_boundary[name] = dict(phi_normal_rms=float(np.sqrt(np.mean(rms(physical_normal_phi, dense_w*mask)**2))),
                                         mu_normal_rms=float(np.sqrt(np.mean(rms(physical_normal_mu, dense_w*mask)**2))))
    special_times = jnp.asarray([.5, 3., 6.], dtype=c.train.compute_dtype)
    bv, adx, ady = [np.asarray(z, dtype=np.float64) for z in eqx.filter_jit(
        lambda n, e, x, t, a: spatial_field_jets(n, e, x, t, a, c))(net, encoded,
        jnp.asarray(dense_b[:, :2], dtype=c.train.compute_dtype), special_times, data['amplitudes'])]
    stress = np.stack((-bv[..., 4] + 2*c.physical.viscosity*adx[..., 2],
                       c.physical.viscosity*(ady[..., 2] + adx[..., 3])), axis=-1)
    outlet = dense_b[:, 4] == 2
    continuous_boundary['traction_times'] = [.5, 3., 6.]
    continuous_boundary['outlet_traction_rms'] = np.sqrt(np.sum(stress**2*dense_w[None, :, None]*outlet[None, :, None], axis=(1, 2))/np.sum(dense_w*outlet)).tolist()
    hard_wall = float(np.max(np.abs(dense_fields[:, dense_b[:, 4] == 0, 2:4])))
    area = (1 - physical[..., 0])/2@np.asarray(raw['volumes'], dtype=np.float64)
    ordinary_loss = np.mean([row['regions']['all_interior']['scaled_mse'][:4] for row in rows], axis=0)
    compare_indices = np.array([20, 60, 120, 240])
    batch = jax.tree.map(lambda v: v[None], data)
    _, canonical = eqx.filter_jit(make_loss_fn(c))(net, batch,
        jnp.asarray(times[compare_indices], dtype=c.train.compute_dtype))
    numerical = np.r_[np.mean([rows[i - 1]['regions']['all_interior']['scaled_mse'][:4] for i in compare_indices], axis=0),
                      np.mean(np.asarray(trained['scaled_per_time'])[compare_indices - 1], axis=0)]
    loss_gap = float(np.max(np.abs(np.asarray(canonical) - numerical)))
    if loss_gap > 3e-6:
        raise ValueError(f'Independent dense-residual evaluation differs from canonical loss: {loss_gap}')
    report = dict(label=label, amplitudes=np.asarray(raw['amplitudes']).tolist(), times=times[1:].tolist(),
        full_BDF_time_nodes=240, full_domain_nodes=n, frozen_primal_gap=primal_gap,
        independent_vs_canonical_loss_maximum_gap=loss_gap, PDE_names=['CH_transport', 'chemical', 'NS_x', 'NS_y', 'GMLS_divergence_diagnostic'],
        mean_PDE_scaled_mse=ordinary_loss.tolist(), BC_names=['wall_phi_n', 'wall_mu_n', 'outlet_traction', 'outlet_phi_n', 'outlet_mu_n'],
        trained_boundary=trained, independent_dense_boundary=dense, continuous_AD_boundary=continuous_boundary,
        maximum_hard_wall_speed=hard_wall, minority_area=area.tolist(), rows=rows,
        arithmetic='Actual FP32 frozen network outputs and stored weights; sparse residual arithmetic accumulated in FP64, checked against actual canonical FP32 loss at four times',
        scope='Frozen nominal development geometry or explicitly selected final-pool training input; no evolved labels/optimization or population claim')
    (output/f'{label}-dense.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    np.savez_compressed(output/f'{label}-residual-maps.npz', residuals=np.asarray(maps),
        points=raw['points'], time=np.array([.5, 1.5, 3., 4.5, 6.]))
    print(json.dumps({'stage': 'dense_audit_complete', 'case': label, 'PDE': ordinary_loss.tolist(),
                      'BC': trained['scaled_mse'], 'canonical_gap': loss_gap}), flush=True)
    if reintegrate:
        rollout_ch(c, coarse, coarse_fields, times, output)
    return report


def rollout_ch(c, raw, learned, times, output):
    scheme = FrozenCH(c, raw, (96, 40))
    with np.load(Path(c.reference.root)/'bdf-fine.npz', allow_pickle=False) as z:
        reference = {k: z[k] for k in z.files}
    query = reference['points']
    bottom, top = walls(query[:, 0], raw['amplitudes'], c, xp=np)
    mapped = np.column_stack((query[:, 0], (query[:, 1] - bottom)/(top - bottom)))
    axes = (np.linspace(0, c.geometry.length, 97), np.linspace(0, 1, 41))

    def export(z):
        return RegularGridInterpolator(axes, np.column_stack((z[:scheme.n], z[scheme.n:])).reshape(97, 41, 2))(mapped)

    results = {}
    for name in ('target_zero_neumann', 'learned_continuous_neumann'):
        old = scheme.initial(); older = old.copy()
        snapshots, rows = [export(old)], []
        started = time.perf_counter()
        for step in range(1, len(times)):
            actual = learned[step].astype(np.float64)
            natural = np.zeros((scheme.n, 2))
            if name == 'learned_continuous_neumann':
                natural[:, 0] = actual[:, 5]*scheme.normal[:, 0] + actual[:, 6]*scheme.normal[:, 1]
                natural[:, 1] = actual[:, 7]*scheme.normal[:, 0] + actual[:, 8]*scheme.normal[:, 1]
            history = old[:scheme.n] if step == 1 else 2*old[:scheme.n] - .5*older[:scheme.n]
            new, info = scheme.step(old, history, actual[:, 2:4], natural,
                                   1. if step == 1 else 1.5, c.time_discretization.dt, 1e-9, 12)
            rows.append(dict(step=step, time=float(times[step]), **info))
            older, old = old, new
            if step % 20 == 0:
                snapshots.append(export(old))
                print(json.dumps({'stage': 'CH_reintegration', 'BC': name, 'step': step,
                                  'seconds': time.perf_counter() - started, **info}), flush=True)
        values = np.asarray(snapshots)
        target = reference['fields'][..., 0]
        relative = lambda a, b: float(np.linalg.norm(a - b)/np.linalg.norm(b))
        record = dict(seconds=time.perf_counter() - started, maximum_scaled_residual=max(r['maximum_scaled_residual'] for r in rows),
            phi_L2=relative(values[..., 0], target), terminal_phi_L2=relative(values[-1, :, 0], target[-1]),
            terminal_minority_L2=relative((1 - values[-1, :, 0])/2, (1 - target[-1])/2))
        np.savez_compressed(output/f'CH-{name}.npz', fields=values, points=query,
            time=reference['time'], reference=reference['fields'][..., :2], amplitudes=raw['amplitudes'])
        (output/f'CH-{name}-steps.json').write_text(json.dumps(rows, indent=2, allow_nan=False) + '\n')
        results[name] = record
        print(json.dumps({'stage': 'CH_complete', 'BC': name, **record}), flush=True)
    with np.load(output/'CH-target_zero_neumann.npz') as a, np.load(output/'CH-learned_continuous_neumann.npz') as b:
        results['BC_change_terminal_phi_L2'] = float(np.linalg.norm(a['fields'][-1, :, 0] - b['fields'][-1, :, 0])/np.linalg.norm(a['fields'][-1, :, 0]))
    results['scope'] = ('96x40/dt.025 CH-only diagnostic, identical frozen learned velocity and analytic IC; '
        'one run zero wall/outlet Neumann, one actual continuous learned normal data; original inlet phi1/mu0. '
        'Not a fully coupled solver or exact production192x80 discretization ablation.')
    (output/'CH-comparison.json').write_text(json.dumps(results, indent=2, allow_nan=False) + '\n')


def main(run, output, *, residual_only=False):
    run, output = Path(run), Path(output)
    output.mkdir(exist_ok=False)
    if jax.default_backend() != 'cpu':
        raise ValueError('Use JAX_PLATFORMS=cpu; preserve independent runtime work')
    c = ConfigDict(json.loads((run/'config.json').read_text())); validate_config(c)
    jax.config.update('jax_enable_x64', False)
    jax.config.update('jax_default_matmul_precision', c.train.matmul_precision)
    key, _ = jax.random.split(jax.random.PRNGKey(c.train.seed))
    net = eqx.tree_deserialise_leaves(run/'model_final.eqx', build_network(c, key))
    times = np.arange(241, dtype=np.float64)*c.time_discretization.dt
    with np.load(run/'evaluation_inputs.npz', allow_pickle=False) as z:
        raw = {k: z[k] for k in z.files}
    audit_case(net, raw, c, times, output, 'nominal', reintegrate=not residual_only)
    with np.load(run/'known_inputs_final.npz', allow_pickle=False) as z:
        raw = {k: z[k][0] for k in z.files}
    audit_case(net, raw, c, times, output, 'final_training_pool_slot0', reintegrate=False)
    (output/'completed.json').write_text(json.dumps(dict(status='completed', run=str(run),
        actual_steps=c.train.iterations, dense_cases=['nominal', 'final_training_pool_slot0'],
        CH_counterfactuals=0 if residual_only else 2), indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--residual-only', action='store_true', help='Skip the separate CH reintegration controls')
    a = parser.parse_args()
    main(a.run, a.output, residual_only=a.residual_only)
