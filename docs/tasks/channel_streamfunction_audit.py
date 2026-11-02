"""Audit a saved streamfunction's AD curl and its GMLS compatibility on CPU."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from ml_collections import ConfigDict
from examples.channel_nsch2d.data import geometry_data
from examples.channel_nsch2d.network import (
    build_network, encode, fields, potentials, spatial_field_jets,
)
from examples.channel_nsch2d.problem import validate_config


def audit(run, step, output):
    run, output = Path(run), Path(output)
    if output.exists():
        raise FileExistsError(output)
    if jax.default_backend() != 'cpu':
        raise ValueError('Run this saved-field diagnostic with JAX_PLATFORMS=cpu')
    c = ConfigDict(json.loads((run / 'config.json').read_text()))
    validate_config(c)
    if c.velocity.kind != 'streamfunction' or c.derivatives.kind != 'gmls_first':
        raise ValueError('Expected the explicit GMLS-first streamfunction control')
    jax.config.update('jax_enable_x64', True)
    jax.config.update('jax_default_matmul_precision', c.train.matmul_precision)
    with np.load(run / f'partial_{step:06d}.npz', allow_pickle=False) as f:
        saved = {k: f[k] for k in f.files}
    raw, _ = geometry_data(saved['amplitudes'], c)
    data = {k: jnp.asarray(v, dtype=v.dtype if np.issubdtype(v.dtype, np.integer)
                          else np.dtype(c.train.compute_dtype)) for k, v in raw.items()}
    key, _ = jax.random.split(jax.random.PRNGKey(c.train.seed))
    net = eqx.tree_deserialise_leaves(run / f'ckpt_{step:06d}.eqx', build_network(c, key))
    times = np.array([.5, 1.5, 3., 4.5, 6.])
    # Mix spatially spread points with interface points at all diagnostic times.
    ids = list(np.linspace(0, len(saved['points']) - 1, 32, dtype=int))
    for t in times:
        ti = np.flatnonzero(np.isclose(saved['time'], t, atol=1e-10, rtol=0))
        if len(ti) != 1:
            raise ValueError(f'Missing exact diagnostic time {t}')
        ids.extend(np.argsort(np.abs(saved['fields'][ti[0], :, 0]))[:12].tolist())
    probe = saved['points'][np.unique(ids)]
    evaluate = eqx.filter_jit(lambda n, e, x, t, a: fields(n, e, x, t, a, c))
    potential = eqx.filter_jit(lambda n, e, x, t, a: potentials(n, e, x, t, a, c))
    jets = eqx.filter_jit(lambda n, e, x, t, a: spatial_field_jets(n, e, x, t, a, c))
    result = {'step': step, 'checkpoint': str(run / f'ckpt_{step:06d}.eqx'),
              'times': times.tolist(), 'probe_count': len(probe), 'precision': {}}
    for dtype, widths in [('float32', [1e-2, 3e-3, 1e-3, 3e-4]),
                          ('float64', [1e-3, 3e-4, 1e-4, 3e-5])]:
        model = jax.tree.map(lambda x: x.astype(dtype) if eqx.is_inexact_array(x) else x, net)
        dd = {k: v.astype(dtype) if eqx.is_inexact_array(v) else v for k, v in data.items()}
        encoded = eqx.filter_jit(encode)(model, dd)
        x, t, a = jnp.asarray(probe, dtype=dtype), jnp.asarray(times, dtype=dtype), dd['amplitudes']
        actual = np.asarray(evaluate(model, encoded, x, t, a))
        _, dx, dy = jets(model, encoded, x, t, a)
        divergence = np.asarray(dx[..., 2] + dy[..., 3], dtype=np.float64)

        def reverse_curl(n, e, x, t, a):
            single = lambda point, time: potentials(n, e, point[None], time[None], a, c)[0, 0, 2]
            derivative = jax.vmap(lambda time: jax.vmap(jax.grad(single), in_axes=(0, None))(x, time))(t)
            return jnp.stack((derivative[..., 1], -derivative[..., 0]), axis=-1)

        independent = np.asarray(eqx.filter_jit(reverse_curl)(model, encoded, x, t, a))
        finite = []
        for h in widths:
            ex, ey = jnp.array([h, 0.], dtype=dtype), jnp.array([0., h], dtype=dtype)
            px = np.asarray(potential(model, encoded, x + ex, t, a))[..., 2]
            mx = np.asarray(potential(model, encoded, x - ex, t, a))[..., 2]
            py = np.asarray(potential(model, encoded, x + ey, t, a))[..., 2]
            my = np.asarray(potential(model, encoded, x - ey, t, a))[..., 2]
            estimate = np.stack(((py - my) / (2 * h), -(px - mx) / (2 * h)), axis=-1)
            error = estimate.astype(np.float64) - actual[..., 2:4]
            finite.append({'h': h, 'maximum_absolute_gap': float(np.max(np.abs(error))),
                           'relative_velocity_L2': float(np.linalg.norm(error) / np.linalg.norm(actual[..., 2:4]))})
        alone = np.asarray(evaluate(model, encoded, x[:1], t, a))
        entry = {'reverse_AD_maximum_gap': float(np.max(np.abs(independent - actual[..., 2:4]))),
                 'query_batch_invariance_maximum_gap': float(np.max(np.abs(alone[:, 0] - actual[:, 0]))),
                 'continuous_divergence_maximum': float(np.max(np.abs(divergence))),
                 'finite_differences': finite}
        result['precision'][dtype] = entry
        if dtype == 'float32':
            actual32 = actual
        else:
            result['promoted64_vs_actual32_velocity_relative_L2'] = float(
                np.linalg.norm(actual[..., 2:4] - actual32[..., 2:4]) / np.linalg.norm(actual[..., 2:4]))
        print(json.dumps({'stage': dtype, **entry}), flush=True)

    encoded = eqx.filter_jit(encode)(net, data)
    t = jnp.asarray(times, dtype=c.train.compute_dtype)

    def full_fields(n, e, x, t, a):
        fn = lambda q: potentials(n, e, q, t, a, c)
        value, px = jax.jvp(fn, (x,), (jnp.broadcast_to(jnp.array([1., 0.], dtype=x.dtype), x.shape),))
        _, py = jax.jvp(fn, (x,), (jnp.broadcast_to(jnp.array([0., 1.], dtype=x.dtype), x.shape),))
        return fields(n, e, x, t, a, c), value[..., 2], px[..., 0], py[..., 0]

    value, psi, phix, phiy = (np.asarray(v, dtype=np.float64) for v in
        eqx.filter_jit(full_fields)(net, encoded, data['points'], t, data['amplitudes']))
    index = np.asarray(data['neighbors'])
    gx, gy = np.asarray(data['gradient_x'], dtype=np.float64), np.asarray(data['gradient_y'], dtype=np.float64)

    def gradient(values):
        adjacent = values[:, index] - values[:, :, None]
        return np.sum(gx[None] * adjacent, axis=-1), np.sum(gy[None] * adjacent, axis=-1)

    phi, u, v = value[..., 0], value[..., 2], value[..., 3]
    ux, _ = gradient(u); _, vy = gradient(v)
    div = ux + vy
    fx, _ = gradient(phi * u); _, fy = gradient(phi * v)
    cx, cy = gradient(phi)
    conservative = fx + fy
    advective = u * cx + v * cy
    exact_advective = u * phix + v * phiy
    product_defect = conservative - advective - phi * div
    psix, psiy = gradient(psi)
    curlx, _ = gradient(psiy); _, curly = gradient(-psix)
    weights = np.asarray(data['transport_measure'], dtype=np.float64) * np.asarray(data['interior'])
    if np.sum(weights) <= 0:
        raise ValueError('Positive interior measure required')

    def rms(a, mask):
        w = weights * mask
        if not np.any(w):
            raise ValueError('Diagnostic region has no interior measure')
        return float(np.sqrt(np.sum(w * a * a) / np.sum(w)))

    rows = []
    for i, time in enumerate(times):
        regions = {}
        for name, mask in [('all_interior', np.ones_like(phi[i], dtype=bool)),
                           ('predicted_interface', np.abs(phi[i]) < .9),
                           ('predicted_bulk', np.abs(phi[i]) > .99)]:
            regions[name] = {label: rms(values[i], mask) for label, values in [
                ('GMLS_divergence_rms', div), ('phi_divergence_rms', phi * div),
                ('GMLS_conservative_transport_rms', conservative),
                ('AD_advective_transport_rms', exact_advective),
                ('conservative_minus_AD_advective_rms', conservative - exact_advective),
                ('GMLS_product_rule_defect_rms', product_defect),
                ('GMLS_curl_discrete_divergence_rms', curlx + curly)]}
        rows.append({'time': float(time), 'regions': regions,
                     'maximum_interior_GMLS_divergence': float(np.max(np.abs(div[i, weights > 0])))})
    result['full_grid_nodes'] = len(value[0])
    result['discrete_compatibility'] = rows
    result['scope'] = ('Saved nominal10k model only; FD/reverse AD verify curl implementation. '
        'GMLS compatibility and product defects describe the learned field, not causal training ablations. '
        'Float64 promotes saved FP32 coefficients solely as a rounding diagnostic. No model/loss/BC/optimizer change.')
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': 'completed', 'report': str(output), 'terminal': rows[-1]}), flush=True)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True)
    p.add_argument('--step', required=True, type=int)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    audit(a.run, a.step, a.output)
