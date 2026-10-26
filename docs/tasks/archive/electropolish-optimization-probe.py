"""Frozen pilot drive/residual/gradient audit; CPU only, no optimizer updates."""
import json
from pathlib import Path
import equinox as eqx
import jax
import jax.numpy as jnp
from ml_collections import ConfigDict
import numpy as np
from scipy import sparse
from examples.corrosion2d import residual_experiment as r
from examples.corrosion2d import residuals
from examples.corrosion2d.configs.residual_pilot import get_config
from nonlocal_pinn.operators import build_D


def main():
    if jax.default_backend() != 'cpu':
        raise ValueError('Set JAX_PLATFORMS=cpu')
    c = get_config('electropolish2d', 'phase', 42)
    c.experiment.initial_sampling = 'matched_graph'
    c.experiment.boundary_sampling = 'matched_graph'
    ref = r.load_reference(c)
    X, Y, points = r.grid(c)
    x = jnp.asarray(points, dtype=jnp.float32)
    op = build_D(x, c.nl.delta, method=c.nl.method, kernel=c.nl.kernel)
    ids, sub, mask, _ = op.build_subgraph_pool(c.sample.K_core, c.sample.K_pool)[0]
    zero, one, _ = r.boundary_values(c, points)
    free = ~(zero | one)
    mask = mask & jnp.asarray(free)[ids]
    xi, ic = r.training_initial_data(c, points, points[:1])
    xz, xo = r.training_boundary_data(c, points, points[zero], points[one])
    kn, kt = jax.random.split(jax.random.PRNGKey(42))
    template = r.build_model(c, kn)
    times = r.sample_training_times(c, kt, 0)
    nd = r.nondim(c)
    row, col, weight = (ref[k] for k in ('graph_row', 'graph_col', 'graph_weight'))
    n = len(points)
    matrix = sparse.diags(np.bincount(row, weights=weight, minlength=n)) - sparse.csr_matrix((weight, (row, col)), shape=(n, n))
    rp, rc = ref['phi'].reshape(21, n), ref['c'].reshape(21, n)
    rq = rc - nd[-2] * (3 * rp**2 - 2 * rp**3) - nd[-1]
    root = Path('/root/autodl-tmp/tf-logs/electropolish2d')
    out = {'scope': 'Frozen old pilot final checkpoints; matched-node gradient probes, no training or causal attribution', 'runs': {}}
    for form in ('phase', 'latent'):
        run = root / f'20261024-residual-{form}-seed42'
        net = eqx.tree_deserialise_leaves(run / 'model_final.eqx', template)

        @eqx.filter_jit
        def fields(m):
            raw = jax.vmap(lambda t: jax.vmap(m.net, in_axes=(0, None))(x, t))(jnp.asarray(ref['time'], dtype=jnp.float32))
            f = m.constraint(raw)
            _, ft = jax.jvp(lambda ts: jax.vmap(lambda t: jax.vmap(m, in_axes=(0, None))(x, t))(ts),
                           (jnp.ones(1),), (jnp.ones(1),))
            return raw, f, ft[0]

        raw, f, ft = (np.asarray(a) for a in fields(net))
        p, conc = f[..., 0].astype(float), f[..., 1].astype(float)
        q = .5 * nd[-1] * (np.tanh(raw[..., 1].astype(float)) - 1)
        dq, rdq = (matrix @ q[-1]), (matrix @ rq[-1])
        ch = ft[:, 1] + nd[0] * dq
        rel = lambda a, b: float(np.linalg.norm(a - b) / np.linalg.norm(b))
        rms = lambda a: float(np.sqrt(np.mean(a*a)))
        drive = {'terminal_q_relative_l2_free': rel(q[-1, free], rq[-1, free]),
                 'terminal_Dq_relative_l2_free': rel(dq[free], rdq[free]),
                 'terminal_stable_CH_rms_free': rms(ch[free]),
                 'initial_q_rmse_over_cle_free': rms((q[0] - rq[0])[free]) / nd[-1],
                 'solid_s_rmse': rms(q[:, one] / nd[-1]),
                 'electrolyte_s_rmse': rms(q[:, zero] / nd[-1] + 1),
                 'terminal_cL_over_cle_range': [float((q[-1]/nd[-1] + 1).min()), float((q[-1]/nd[-1] + 1).max())]}
        probes = {}
        for phase, mode in (('phase', 'physical'), ('phase', 's_bc'), ('latent', 's_bc')):
            setting = c.copy_and_resolve_references()
            setting.experiment.formulation, setting.experiment.data_loss_mode = phase, mode
            loss, _ = r.make_training_losses(setting, nd, xi, ic, xz, xo)

            @eqx.filter_jit
            def gradients(m):
                values, back = eqx.filter_vjp(lambda a: loss(a, times, x[ids], sub, mask), m)
                norms = jnp.stack([sum(jnp.sum(jnp.abs(a)) for a in jax.tree.leaves(back(jax.nn.one_hot(i, 5))[0]) if eqx.is_array(a)) for i in range(5)])
                clipped = jnp.clip(norms, 1e-6, 1e6)
                weights = jnp.clip(jnp.mean(clipped) / clipped, 1e-6, setting.loss.gradnorm_weight_cap)
                return values, norms, weights

            values, norms, weights = (np.asarray(a) for a in gradients(net))
            probes[phase + '/' + mode] = {'losses': values.tolist(), 'parameter_gradient_L1': norms.tolist(), 'GradNorm_proposals': weights.tolist()}
        out['runs'][form] = {'path': str(run), 'drive': drive, 'loss_order': ['AC', 'CH', 'zero_BC', 'one_BC', 'IC'], 'frozen_gradient_probes': probes}
        print(form, json.dumps(out['runs'][form]), flush=True)
    Path(__file__).with_suffix('.json').write_text(json.dumps(out, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
