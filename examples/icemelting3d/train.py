"""3D PINN for ice melting — nonlocal Allen-Cahn with analytical solution.

Usage:
  source .venv/bin/activate
  python examples/icemelting3d/train.py --config=examples/icemelting3d/configs/train_base.py
"""

import importlib.util, logging, os, time
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from tensorboardX import SummaryWriter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from absl import app as _absl_app
from ml_collections import config_flags as _config_flags

from pinn.operators import build_D, SparseD
from pinn.arch import make_pinn, make_film_pinn
from pinn.arch.hard_constraint import TanhConstraint
from pinn.optimizers import make_optimizer

log = logging.getLogger(__name__)


# ── Setup helpers ──────────────────────────────────────────────────────────

def _load_losses():
    spec = importlib.util.spec_from_file_location(
        "losses", os.path.join(os.path.dirname(__file__), "losses.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _setup_dirs(config):
    run = config.log.run_name or time.strftime('%Y%m%d-%H%M%S')
    run_dir = os.path.join(config.log.log_dir, run)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


# ── Network ────────────────────────────────────────────────────────────────

def _build_network(config, key):
    arch = config.arch
    hc = TanhConstraint() if arch.use_hard_constraint else None
    activation = getattr(jnp, arch.activation) if hasattr(jnp, arch.activation) else jnp.tanh

    common = dict(
        spatial_dim=arch.spatial_dim,
        out_dim=arch.out_dim,
        hidden_dim=arch.hidden_dim,
        num_layers=arch.num_layers,
        spatial_emb_dim=arch.spatial_emb_dim,
        spatial_scale=arch.spatial_scale,
        temporal_emb_dim=arch.temporal_emb_dim,
        temporal_scale=arch.temporal_scale,
        activation=activation,
        hard_constraint=hc,
        key=key,
    )

    if arch.type == "film":
        return make_film_pinn(
            film_hidden_dim=arch.film_hidden_dim,
            film_num_layers=arch.film_num_layers,
            **common,
        )
    else:
        return make_pinn(trunk_type=arch.type, **common)


# ── Non-dimensional parameters ─────────────────────────────────────────────

def _nondim_params(config):
    phys = config.physical
    nd = config.nondim
    L_c = nd.L_c
    T_c = nd.T_c

    N = config.sample.Nx
    h_dim = 100.0 / N
    eps_dim = 6.0 * h_dim / (2.0 * np.sqrt(2.0) * np.arctanh(0.9))
    eps_star = eps_dim / L_c

    R0_star = phys.R0 / L_c
    lam_star = phys.lam * T_c / L_c
    N1 = phys.M * T_c / L_c**2
    N2 = N1 / eps_star**2
    N3 = lam_star / (np.sqrt(2.0) * eps_star)
    denom = np.sqrt(2.0) * eps_star

    return N1, N2, N3, eps_star, denom, lam_star, R0_star


# ── Training step ──────────────────────────────────────────────────────────

def _make_step(loss_fn, optim):
    @eqx.filter_jit
    def step(net, opt_state, t_batch, phi0, w_ema, bias, update_w,
             iface_gamma, x_points, sub_data, core_mask):
        (loss, aux, grad, new_w) = loss_fn(
            net, t_batch, phi0, w_ema, bias, update_w, iface_gamma,
            x_points, sub_data, core_mask)
        grad = jax.tree.map(jnp.nan_to_num, grad)
        updates, opt_state = optim.update(grad, opt_state, net)
        net = eqx.apply_updates(net, updates)
        return net, opt_state, loss, aux, new_w
    return step


# ── Evaluation ─────────────────────────────────────────────────────────────

def _eval_analytic(net, x_eval, t_eval, R0_star, lam_star, xc, denom,
                   writer, step):
    losses = _load_losses()

    def _phi_at_t(t):
        return jax.vmap(net, in_axes=(0, None))(x_eval, t)[:, 0]

    phi_pred = jax.vmap(_phi_at_t)(t_eval)
    phi_anal = jax.vmap(lambda t: losses.analytic_solution(
        x_eval, t, R0_star, lam_star, xc, denom))(t_eval)

    err = jnp.sqrt(jnp.mean((phi_pred - phi_anal)**2, axis=1))
    norm = jnp.sqrt(jnp.mean(phi_anal**2, axis=1))
    rel = err / jnp.maximum(norm, 1e-12)

    for i, t in enumerate(t_eval):
        writer.add_scalar('eval/rel_L2', float(rel[i]), step)
    writer.add_scalar('eval/rel_L2_mean', float(jnp.mean(rel)), step)
    return float(jnp.mean(rel))


# ── Visualisation ──────────────────────────────────────────────────────────

def _plot_snapshots(net, x_slice, R0_star, lam_star, xc, denom,
                    writer, step, Nx, Ny):
    """Plot φ snapshots at key times on a 2D slice (z=0.5)."""
    losses = _load_losses()
    target_times = [0.0, 0.25, 0.5, 0.75, 1.0]
    t_arr = jnp.array(target_times)

    # Predicted on slice for all t
    phi_pred_all = jax.vmap(
        lambda t: jax.vmap(net, in_axes=(0, None))(x_slice, t)[:, 0])(t_arr)
    phi_anal_all = jax.vmap(
        lambda t: losses.analytic_solution(
            x_slice, t, R0_star, lam_star, xc, denom))(t_arr)
    err_all = jnp.abs(phi_pred_all - phi_anal_all)
    max_err = float(jnp.max(err_all))

    n = len(target_times)
    fig, axes = plt.subplots(2, n, figsize=(3 * n, 5), squeeze=False)
    for j in range(n):
        phi_p = np.array(phi_pred_all[j]).reshape(Ny, Nx)
        err = np.array(err_all[j]).reshape(Ny, Nx)
        im0 = axes[0][j].pcolormesh(phi_p, cmap='coolwarm', vmin=-1, vmax=1,
                                     shading='auto')
        axes[0][j].set_title(rf'$\phi$ PINN  t*={target_times[j]:.2f}')
        axes[0][j].set_aspect('equal')

        im1 = axes[1][j].pcolormesh(err, cmap='coolwarm', vmin=0, vmax=max_err,
                                     shading='auto')
        axes[1][j].set_title(
            rf'$|\phi - \phi_{{anal}}|$  t*={target_times[j]:.2f}')
        axes[1][j].set_aspect('equal')

    fig.tight_layout(rect=[0, 0.05, 1, 1])
    cbar_ax = fig.add_axes([0.15, 0.02, 0.7, 0.02])
    fig.colorbar(im1, cax=cbar_ax, orientation='horizontal', label='|error|')
    writer.add_figure('eval/snapshots', fig, step)
    plt.close(fig)


# ── Diagnostic ─────────────────────────────────────────────────────────────

def _diag_residual(net, x_eval, D, t_eval, N1, N2, N3, writer, step):
    losses = _load_losses()
    phi, dphi_dt = losses._compute_all(net, x_eval, t_eval)
    D_phi = D @ phi

    components = {
        'dphi_dt': dphi_dt,
        'N1_Dphi': N1 * D_phi,
        'N2_cubic': N2 * (phi**3 - phi),
        'N3_source': N3 * (1.0 - phi**2),
    }
    for name, val in components.items():
        writer.add_scalar(f'diag/{name}_rms',
                          float(jnp.sqrt(jnp.mean(val**2))), step)
    writer.add_scalar('diag/total_rms',
                      float(jnp.sqrt(jnp.mean(sum(components.values())**2))), step)


# ── Main ───────────────────────────────────────────────────────────────────

def _main(_):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    config = FLAGS.config
    run_dir = _setup_dirs(config)
    writer = SummaryWriter(run_dir)
    log.info("Logging to %s", run_dir)

    losses = _load_losses()
    make_loss_fn = losses.make_loss_fn
    ic_values = losses.ic_values

    # ── Grid ────────────────────────────────────────────────────────────
    Nx, Ny, Nz = config.sample.Nx, config.sample.Ny, config.sample.Nz
    xs = jnp.linspace(0.0, 1.0, Nx)
    ys = jnp.linspace(0.0, 1.0, Ny)
    zs = jnp.linspace(0.0, 1.0, Nz)
    Xv, Yv, Zv = jnp.meshgrid(xs, ys, zs, indexing='xy')
    x_base = jnp.stack([Xv.ravel(), Yv.ravel(), Zv.ravel()], axis=-1)
    log.info("Grid: %dx%dx%d = %d points", Ny, Nx, Nz, len(x_base))

    # ── Non-dimensional parameters ──────────────────────────────────────
    N1, N2, N3, eps_star, denom, lam_star, R0_star = _nondim_params(config)
    xc = np.array(config.nondim.xc)
    log.info("N1=%.6e  N2=%.6e  N3=%.6e", N1, N2, N3)
    log.info("eps*=%.4f  R0*=%.4f  lam*=%.4f", eps_star, R0_star, lam_star)

    # ── Network ─────────────────────────────────────────────────────────
    rng = jr.PRNGKey(42)
    k_net, k_ic = jr.split(rng)
    net = _build_network(config, k_net)
    log.info("Network: %s, out_dim=%d, spatial_dim=%d",
             config.arch.type, config.arch.out_dim, config.arch.spatial_dim)

    # ── Nonlocal operator ───────────────────────────────────────────────
    D = build_D(x_base, config.nl.delta,
                kernel=config.nl.kernel, method=config.nl.method)
    if isinstance(D, SparseD):
        log.info("D: SparseD N=%d nnz=%d k_avg=%.1f",
                 D._N, len(D._vals), len(D._vals) / D._N)
    else:
        log.info("D: DenseD shape=%s", D.shape)

    # ── IC points ───────────────────────────────────────────────────────
    idx_ic = jr.choice(k_ic, len(x_base), (config.sample.N_ic,), replace=False)
    x_ic = x_base[idx_ic]
    phi0 = ic_values(x_ic, R0_star, xc, denom)
    log.info("IC: N_ic=%d, phi min=%.3f max=%.3f",
             len(x_ic), float(jnp.min(phi0)), float(jnp.max(phi0)))

    # ── Sub-graph pool ──────────────────────────────────────────────────
    K_core = config.sample.K_core
    subgraph_pool = None
    if K_core > 0 and K_core < len(x_base) and isinstance(D, SparseD):
        K_pool = config.sample.K_pool
        log.info("Pre-building sub-graph pool (%d entries, K_core=%d) ...",
                 K_pool, K_core)
        _t0 = time.perf_counter()
        raw_pool = [D.sub_sample(i, K_core) for i in range(K_pool)]
        _K_alls = [r[3] for r in raw_pool]
        log.info("  K_all: min=%d max=%d avg=%d (K_core=%d, N=%d)",
                 min(_K_alls), max(_K_alls), int(np.mean(_K_alls)),
                 K_core, D._N)
        subgraph_pool = [(x_base[r[0]], r[1], r[2]) for r in raw_pool]
        del raw_pool
        _N_pad = len(subgraph_pool[0][0])
        _E_pad = len(subgraph_pool[0][1][0])
        log.info("Pool ready in %.1fs: N_pad=%d E_pad=%d",
                 time.perf_counter() - _t0, _N_pad, _E_pad)

    # ── Optimizer ───────────────────────────────────────────────────────
    optim = make_optimizer(config)
    opt_state = optim.init(eqx.filter(net, eqx.is_array))

    # ── Loss function ───────────────────────────────────────────────────
    lfn = make_loss_fn(
        x_base, D, N1, N2, N3, x_ic=x_ic,
        alpha=config.train.alpha_gradnorm,
        gradnorm_cap=config.loss.gradnorm_weight_cap,
    )

    # ── Training state ──────────────────────────────────────────────────
    step_fn = _make_step(lfn, optim)
    w_ema = jnp.ones(2)
    bias = jnp.array(config.loss.gradnorm_bias)

    # Eval setup
    N_eval = min(8000, len(x_base))
    idx_eval = jr.choice(jr.PRNGKey(123), len(x_base), (N_eval,), replace=False)
    x_eval = x_base[idx_eval]
    t_eval = jnp.linspace(0.0, 1.0, 11)

    # Slice at z=0.5 for visualisation
    xs_1d = jnp.linspace(0.0, 1.0, Nx)
    ys_1d = jnp.linspace(0.0, 1.0, Ny)
    Xv_s, Yv_s = jnp.meshgrid(xs_1d, ys_1d, indexing='xy')
    x_slice = jnp.stack([Xv_s.ravel(), Yv_s.ravel(),
                         jnp.full(Xv_s.size, 0.5)], axis=-1)

    # ── Training loop ───────────────────────────────────────────────────
    cfg = config.train
    log_cfg = config.log
    t_min = config.sample.t_min

    for it in range(1, cfg.iterations + 1):
        key_t = jr.PRNGKey(it * 3 + 1000)
        t_batch = jr.uniform(key_t, (config.sample.Nt,),
                             minval=t_min, maxval=1.0)
        update_w = (it % cfg.gradnorm_update_every == 1)

        # Interface weight decay
        gamma_decay = config.loss.iface_gamma_decay_steps
        if gamma_decay > 0 and it <= gamma_decay:
            iface_gamma = jnp.array(config.loss.iface_gamma_init
                                    * (1.0 - it / gamma_decay))
        else:
            iface_gamma = jnp.array(config.loss.iface_gamma_decay_to)

        # Spatial minibatch
        if subgraph_pool is not None:
            x_points, sub_data, core_mask = subgraph_pool[it % len(subgraph_pool)]
        else:
            x_points = x_base
            sub_data = core_mask = None

        net, opt_state, loss_val, aux, w_ema = step_fn(
            net, opt_state, t_batch, phi0, w_ema, bias, update_w,
            iface_gamma, x_points, sub_data, core_mask)

        # Logging
        if it % log_cfg.log_every == 0:
            writer.add_scalar('loss/total', float(loss_val), it)
            writer.add_scalar('loss/pde', float(aux['loss_pde']), it)
            writer.add_scalar('loss/ic', float(aux['loss_ic']), it)
            writer.add_scalar('weight/pde', float(aux['weight_pde']), it)
            writer.add_scalar('weight/ic', float(aux['weight_ic']), it)
            log.info("%5d: total=%.4e  pde=%.4e  ic=%.4e  w=[%.2f,%.2f]",
                     it, float(loss_val), float(aux['loss_pde']),
                     float(aux['loss_ic']),
                     float(aux['weight_pde']), float(aux['weight_ic']))

        # Evaluation
        if it % log_cfg.eval_every == 0:
            rel = _eval_analytic(net, x_eval, t_eval,
                                 R0_star, lam_star, xc, denom, writer, it)
            log.info("  --- eval %d: rel_L2=%.6e ---", it, rel)
            _plot_snapshots(net, x_slice, R0_star, lam_star,
                            xc, denom, writer, it, Nx, Ny)

        # Diagnostic
        if log_cfg.run_diag and it % log_cfg.eval_every == 0:
            _diag_residual(net, x_eval, D, jnp.array([0.2, 0.5, 0.8]),
                           N1, N2, N3, writer, it)

        # Checkpoint
        if log_cfg.save_ckpt and it % log_cfg.eval_every == 0:
            ckpt_path = os.path.join(run_dir, f"ckpt_{it:06d}.eqx")
            eqx.tree_serialise_leaves(ckpt_path, net)
            log.info("  Checkpoint saved: %s", ckpt_path)

    # ── Final save ──────────────────────────────────────────────────────
    final_path = os.path.join(run_dir, "model_final.eqx")
    eqx.tree_serialise_leaves(final_path, net)
    log.info("Final model: %s", final_path)
    writer.close()


# ── Config ────────────────────────────────────────────────────────────────

_DEFAULT = os.path.join(os.path.dirname(__file__), 'configs', 'train_base.py')
_CONFIG = _config_flags.DEFINE_config_file('config', _DEFAULT, 'Config file')
FLAGS = _absl_app.flags.FLAGS

if __name__ == '__main__':
    _absl_app.run(_main)
