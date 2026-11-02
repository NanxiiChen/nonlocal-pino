"""Resolve boundary aliasing and GMLS-versus-AD normal derivatives of saved fields."""
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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from examples.channel_nsch2d.network import build_network, encode, potentials
from examples.channel_nsch2d.problem import boundary
from examples.channel_nsch2d.operators import first_derivative_stencils


def inspect(run, output):
    run, output = Path(run), Path(output)
    output.mkdir(exist_ok=False)
    c = ConfigDict(json.loads((run/'config.json').read_text()))
    jax.config.update('jax_enable_x64', False)
    jax.config.update('jax_default_matmul_precision', c.train.matmul_precision)
    if jax.default_backend() != 'cpu':
        raise ValueError('CPU-only saved-model diagnostic required')
    with np.load(run/'evaluation_inputs.npz', allow_pickle=False) as z:
        raw = {k: z[k] for k in z.files}
    with np.load(run/'diagnostic_fields.npz', allow_pickle=False) as z:
        saved = {k: z[k] for k in z.files}
    key, _ = jax.random.split(jax.random.PRNGKey(c.train.seed))
    net = eqx.tree_deserialise_leaves(run/'model_final.eqx', build_network(c, key))
    data = {k: jnp.asarray(v) for k, v in raw.items()}
    encoded = eqx.filter_jit(encode)(net, data)
    cc = ConfigDict(c.to_dict()); cc.sample.boundary_points = 512
    dense, dw = boundary(raw['amplitudes'], cc)
    original, ow = raw['boundary'], raw['boundary_weights']
    both = np.concatenate((original, dense))
    times = np.array([.5, 1.5, 3., 4.5, 6.])

    @eqx.filter_jit
    def derivatives(n, e, x, t, a):
        fn = lambda p: potentials(n, e, p, t, a, c)[..., :2]
        value, dx = jax.jvp(fn, (x,), (jnp.broadcast_to(jnp.array([1., 0.], dtype=x.dtype), x.shape),))
        _, dy = jax.jvp(fn, (x,), (jnp.broadcast_to(jnp.array([0., 1.], dtype=x.dtype), x.shape),))
        return value, dx, dy

    val, dx, dy = [np.asarray(a, dtype=np.float64) for a in derivatives(net, encoded,
        jnp.asarray(both[:, :2], dtype=c.train.compute_dtype), jnp.asarray(times, dtype=c.train.compute_dtype), data['amplitudes'])]
    ad = dx*both[None, :, 2, None] + dy*both[None, :, 3, None]
    native = saved['fields'][np.array([1, 3, 6, 9, 12]), :, :2].astype(np.float64)
    ni, gx, gy, _ = first_derivative_stencils(raw['points'], both[:, :2], c)
    delta = native[:, ni, :] - val[:, :, None, :]
    numerical = np.sum(gx[None, :, :, None]*delta, axis=2)*both[None, :, 2, None]
    numerical += np.sum(gy[None, :, :, None]*delta, axis=2)*both[None, :, 3, None]
    rows = []
    for i, t in enumerate(times):
        values = {}
        for name, selection, b, weight in [('training_sites', slice(0, len(original)), original, ow),
                                          ('independent512', slice(len(original), None), dense, dw)]:
            wall = b[:, 4] == 0
            W = weight*wall
            values[name] = {method: np.sqrt(np.sum(value[i, selection]**2*W[:, None], axis=0)/W.sum()).tolist()
                            for method, value in [('AD', ad), ('GMLS', numerical)]}
        rows.append(dict(time=float(t), normal_rms_phi_mu=values))
    fig, axes = plt.subplots(2, 2, figsize=(12, 6), layout='constrained')
    for col, ti in enumerate((2, 4)):
        for row, side in enumerate(('bottom', 'top')):
            ax = axes[row, col]
            number = len(dense)//4
            # boundary() concatenates bottom, top, inlet, outlet in this order.
            block = slice(row*number, (row + 1)*number)
            train_n = len(original)//4
            train_block = slice(row*train_n, (row + 1)*train_n)
            ax.plot(dense[block, 0], ad[ti, len(original):, 1][block], lw=1.2, label='AD normal derivative')
            ax.plot(dense[block, 0], numerical[ti, len(original):, 1][block], lw=1., alpha=.7, label='GMLS normal derivative')
            ax.scatter(original[train_block, 0], numerical[ti, :len(original), 1][train_block], s=15,
                       color='#D55E00', label='GMLS at training sites', zorder=5)
            ax.axhline(0, color='black', lw=.7)
            ax.set(xlabel='x', ylabel='normal derivative of mu', title=f'{side} wall, t={times[ti]:g}')
            ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle('Actual saved20k boundary traces: low training-site residual can hide unresolved boundary variation')
    fig.savefig(output/'mu-normal-traces.png', dpi=180); fig.savefig(output/'mu-normal-traces.pdf'); plt.close(fig)
    report = dict(status='completed', times=times.tolist(), rows=rows, training_boundary_points_per_side=len(original)//4,
                  independent_boundary_points_per_side=512,
                  scope='Same final model and encoder, original versus independent boundary sites. GMLS reconstructed with FP64 accumulation; no model/BC/training change')
    (output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    np.savez_compressed(output/'traces.npz', times=times, boundary=both, AD=ad, GMLS=numerical)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True); p.add_argument('--output', required=True)
    a = p.parse_args(); inspect(a.run, a.output)
