"""CPU-only derivative candidate; never imported by the live CH trainer.

For d+=(1,1), d-=(1,-1), binomial expansion gives
D_+^4 + D_-^4 = 2 D_x^4 + 12 D_xxyy + 2 D_y^4.
Thus Delta^2 = (2/3)(D_x^4+D_y^4) + (1/6)(D_+^4+D_-^4).
Taylor coefficients are converted to derivatives by exact integer factorials.
This script checks spatial jets and parameter backward, not full CH training.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.experimental.jet import jet
import numpy as np


def taylor_spatial_jets(function, points, fourth):
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError('Explicit two-dimensional independent queries required')
    order = 4 if fourth else 2
    directions = jnp.asarray(((1., 0.), (0., 1.), (1., 1.), (1., -1.)), dtype=points.dtype)
    if not fourth:
        directions = directions[:2]
    zeros = (jnp.zeros_like(points),)*(order-1)

    def along(direction):
        series = (jnp.broadcast_to(direction, points.shape), *zeros)
        _, coefficients = jet(function, (points,), (series,), factorial_scaled=False)
        return tuple(a*k for a, k in zip(coefficients, (1, 2, 6, 24)[:order], strict=True))

    values = jax.vmap(along)(directions)
    gradient = jnp.stack((values[0][0], values[0][1]), axis=-1)
    laplacian = values[1][0]+values[1][1]
    if not fourth:
        return gradient, laplacian
    biharmonic = (2/3)*(values[3][0]+values[3][1])+(1/6)*(values[3][2]+values[3][3])
    return gradient, laplacian, biharmonic


def compare(actual, expected, *, atol, rtol):
    rows = []
    for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        a, b = np.asarray(a), np.asarray(b)
        np.testing.assert_allclose(a, b, atol=atol, rtol=rtol)
        rows.append(float(np.max(np.abs(a-b))) if a.size else 0.)
    return dict(leaves=len(rows), maximum_absolute_difference=max(rows))


def qualify():
    if jax.default_backend() != 'cpu':
        raise RuntimeError('This investigation must run on CPU while the GPU probe continues')
    jax.config.update('jax_enable_x64', True)
    from examples.confined_ch2d.operator_local_ad import raw, spatial_jets
    from examples.confined_ch2d.operator_training import build_network, encode
    from tests.test_confined_ch_operator_local_ad import config, data

    points = jnp.array([[.23, .32], [.54, .66]])
    x, y = np.asarray(points).T
    polynomial = lambda q: q[:, 0]**4+q[:, 0]**2*q[:, 1]**2+2*q[:, 1]**4
    exact = (np.stack((4*x**3+2*x*y*y, 2*x*x*y+8*y**3), axis=-1),
             14*x*x+26*y*y, np.full(len(x), 80.))
    polynomial_check = compare(taylor_spatial_jets(polynomial, points, True), exact,
                               atol=2e-11, rtol=2e-12)
    k = jnp.array([2.3, -3.7]); phase = points@k+.2
    fourier = lambda q: jnp.cos(q@k+.2)
    exact = (-jnp.sin(phase)[:, None]*k, -jnp.sum(k*k)*jnp.cos(phase),
             jnp.sum(k*k)**2*jnp.cos(phase))
    fourier_check = compare(taylor_spatial_jets(fourier, points, True), exact,
                            atol=2e-10, rtol=2e-12)

    c = config(); c.arch.width=4; c.arch.heads=1; c.arch.condition_depth=1
    c.arch.decoder_depth=1; c.arch.spatial_features=2; c.arch.time_features=2
    d = data(c); net = build_network(c, jax.random.key(11)); t = jnp.asarray(.5)
    encoded = encode(net, d)
    f = lambda q: raw(net, encoded, q, t, c)
    network_check = compare(taylor_spatial_jets(f, points, True), spatial_jets(f, points, True),
                            atol=2e-8, rtol=2e-8)
    second_check = compare(taylor_spatial_jets(f, points, False), spatial_jets(f, points, False),
                           atol=2e-9, rtol=2e-9)
    results, timings = {}, {}
    for name, jets in (('nested_jvp', spatial_jets), ('taylor', taylor_spatial_jets)):
        def objective(model):
            context = encode(model, d)
            values = jets(lambda q: raw(model, context, q, t, c), points, True)
            return sum(jnp.mean(value**2)/scale for value, scale in zip(values, (1., 10., 100.), strict=True))
        run = eqx.filter_jit(eqx.filter_value_and_grad(objective))
        start = time.perf_counter(); result = run(net)
        jax.block_until_ready(result); first = time.perf_counter()-start
        start = time.perf_counter(); again = run(net)
        jax.block_until_ready(again); warm = time.perf_counter()-start
        results[name] = result; timings[name] = dict(first_seconds=first, warm_seconds=warm)
        print(json.dumps(dict(method=name, cpu_timing=timings[name])), flush=True)
    gradient_check = compare(results['taylor'], results['nested_jvp'], atol=2e-7, rtol=2e-8)
    return dict(status='passed', created_utc=datetime.now(timezone.utc).isoformat(),
        jax=jax.__version__, backend='cpu', precision='float64', polynomial=polynomial_check,
        fourier=fourier_check, actual_network_jets=network_check, actual_network_second_jets=second_check,
        parameter_backward=gradient_check, timings=timings,
        scope='Width4 one-block two-stream network; equivalent derivative-weighted diagnostic objective, not the full CH loss or a production-GPU benchmark',
        live_training_modified=False, production_sources_modified=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = qualify()
    with args.output.open('x') as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(report, indent=2))
