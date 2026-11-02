"""Frozen-network BDF2 refinement against its analytic physical time rate."""
from pathlib import Path
import argparse
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def probe(run, output):
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import numpy as np
    from ml_collections import ConfigDict
    from nonlocal_pinn.training.runtime import configure_gpu_policy
    from examples.channel_nsch2d.data import geometry_data
    from examples.channel_nsch2d.network import build_network, encode, fields
    from examples.channel_nsch2d.temporal import backward_rate, previous_times

    run = Path(run)
    c = ConfigDict(json.loads((run / 'config.json').read_text()))
    configure_gpu_policy(c.train.gpu_policy)
    jax.config.update('jax_enable_x64', c.train.compute_dtype == 'float64')
    jax.config.update('jax_default_matmul_precision', c.train.matmul_precision)
    raw, _ = geometry_data((c.geometry.upper, c.geometry.lower), c)
    data = {k: jnp.asarray(v, dtype=v.dtype if np.issubdtype(v.dtype, np.integer)
                          else c.train.compute_dtype) for k, v in raw.items()}
    key, _ = jax.random.split(jax.random.PRNGKey(c.train.seed))
    net = eqx.tree_deserialise_leaves(run / 'model_final.eqx', build_network(c, key))
    # Same physical interior sites and times in every rate comparison.
    indices = np.flatnonzero(raw['interior'])[::61]
    points = data['points'][indices]
    times = jnp.asarray([.1, .2, .5, 1., 1.5, 3., 6.], dtype=c.train.compute_dtype)

    @eqx.filter_jit
    def evaluate(model):
        encoded = encode(model, data)
        predict = lambda t: fields(model, encoded, points, t, data['amplitudes'], c)
        current, exact = jax.jvp(predict, (times,), (jnp.ones_like(times),))
        rates = []
        for dt in (.1, .05, .025, .0125):
            t1, t2 = previous_times(times, dt)
            history = predict(jnp.concatenate((t1, t2)))
            rates.append(backward_rate(current, history[:len(times)],
                                      history[len(times):], times, dt))
        return exact, jnp.stack(rates)

    exact, rates = [np.asarray(a, dtype=np.float64) for a in evaluate(net)]
    if not np.isfinite(exact).all() or not np.isfinite(rates).all():
        raise FloatingPointError('Nonfinite physical time consistency probe')
    records = []
    for dt, rate in zip((.1, .05, .025, .0125), rates, strict=True):
        groups = {}
        for label, channels in (('phi', [0]), ('velocity', [2, 3])):
            error = (rate - exact)[..., channels]
            target = exact[..., channels]
            norm = np.linalg.norm(target)
            if norm == 0:
                raise ValueError(f'Zero exact {label} rate norm')
            groups[label] = dict(relative=float(np.linalg.norm(error) / norm),
                rms_by_time=np.sqrt(np.mean(error * error, axis=(1, 2))).tolist(),
                exact_rms_by_time=np.sqrt(np.mean(target * target, axis=(1, 2))).tolist())
        records.append(dict(dt=dt, fields=groups))
    report = dict(status='frozen_time_consistency_completed', run=str(run),
        points=len(indices), times=np.asarray(times, dtype=float).tolist(), records=records,
        scope='BDF error of this frozen differentiable network, not FEM or solution accuracy; no updates')
    with Path(output).open('x') as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    probe(args.run, args.output)
