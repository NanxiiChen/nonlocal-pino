"""CPU-only production-architecture jets; no edits to the live GPU trainer.

Use the actual width/depth/Fourier/time configuration and cold seed42. Only the
number of diagnostic coordinates and encoder points is reduced. This qualifies
forward derivatives, not a production training step or parameter backward.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from docs.tasks.spinodal_taylor_ad_probe import taylor_spatial_jets
from examples.confined_ch2d.configs.train_operator_compare_local_ad import get_config
from examples.confined_ch2d.operator_local_ad import raw, spatial_jets
from examples.confined_ch2d.operator_precision import compute_arrays, compute_dtype
from examples.confined_ch2d.operator_training import build_network, encode
from tests.test_confined_ch_operator_local_ad import data


def qualify(precision):
    if jax.default_backend() != 'cpu':
        raise RuntimeError('Only a CPU diagnostic may coexist with the live GPU probe')
    jax.config.update('jax_enable_x64', True)
    jax.config.update('jax_default_matmul_precision', 'highest')
    c = get_config()
    c.train.compute_dtype = precision
    c.geometry.boundary_outer = 4
    c.geometry.boundary_per_hole = 2
    d = compute_arrays(data(c), c)
    net = build_network(c, jax.random.key(42))
    encoded = eqx.filter_jit(encode)(net, d)
    jax.block_until_ready(encoded)
    times = jnp.asarray([.3, .7], dtype=compute_dtype(c))
    outputs, timings = {}, {}
    for name, jets in (('taylor', taylor_spatial_jets), ('nested_jvp', spatial_jets)):
        @eqx.filter_jit
        def evaluate(model, context, points, ts):
            return jax.vmap(lambda t: jets(lambda x: raw(model, context, x, t, c), points, True))(ts)

        start = time.perf_counter()
        result = evaluate(net, encoded, d['points'], times)
        jax.block_until_ready(result)
        first = time.perf_counter()-start
        start = time.perf_counter()
        jax.block_until_ready(evaluate(net, encoded, d['points'], times))
        warm = time.perf_counter()-start
        outputs[name] = tuple(np.asarray(a) for a in result)
        timings[name] = dict(first_seconds=first, warm_seconds=warm)
        print(json.dumps(dict(method=name, timing=timings[name])), flush=True)

    # FP32 tolerances are declared before observing the production values.
    atol, rtol = (1e-7, 2e-8) if precision == 'float64' else (1e-3, 3e-4)
    checks = {}
    for index, (label, actual, expected) in enumerate(zip(
            ('gradient', 'laplacian', 'biharmonic'),
            outputs['taylor'], outputs['nested_jvp'], strict=True)):
        if actual.dtype != compute_dtype(c) or expected.dtype != compute_dtype(c):
            raise AssertionError('Unexpected precision promotion')
        if not np.isfinite(actual).all() or not np.isfinite(expected).all():
            raise AssertionError('Nonfinite forward derivatives')
        np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol)
        difference = np.asarray(actual, dtype=np.float64)-expected
        denominator = float(np.linalg.norm(expected.astype(np.float64)))
        if denominator <= 0:
            raise AssertionError('Production derivative norm must be positive')
        checks[label] = dict(maximum_absolute_difference=float(np.max(np.abs(difference))),
                            relative_norm=float(np.linalg.norm(difference)/denominator),
                            values={k: v[index].tolist() for k, v in outputs.items()})
    return dict(status='passed', created_utc=datetime.now(timezone.utc).isoformat(),
                precision=precision, backend='cpu', architecture=c.arch.to_dict(), seed=42,
                tolerance=dict(atol=atol, rtol=rtol), jax=jax.__version__,
                timings=timings, checks=checks, domain_queries=len(d['points']),
                wall_queries=len(d['condition']), times=np.asarray(times).tolist(),
                scope='Actual production architecture; reduced encoder/query count; forward first/second/fourth derivatives',
                limitation='Not full-grid GPU resource or complete parameter-backward qualification',
                live_training_modified=False, production_sources_modified=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--precision', choices=('float32', 'float64'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = qualify(args.precision)
    with args.output.open('x') as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(report, indent=2))
