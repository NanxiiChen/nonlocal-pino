"""Training script for split-PINN on 1D KKS phase-field corrosion.

Usage (from project root):
  python examples/corrosion1d_activation/train.py
  python examples/corrosion1d_activation/train.py --config=my_config.py
  python examples/corrosion1d_activation/train.py --config.train.iterations=1000 --config.nl.delta=0.05
"""

import logging, os, time

import jax, jax.numpy as jnp, jax.random as jr
import equinox as eqx
import numpy as np
from tensorboardX import SummaryWriter

log = logging.getLogger(__name__)

from pinn.operators import build_D, DWithBC
from pinn.arch import make_split_pinn

# Import losses from sibling file (example is not a package)
import importlib.util as _iu
_losses_spec = _iu.spec_from_file_location(
    "losses", os.path.join(os.path.dirname(__file__), 'losses.py'))
_losses = _iu.module_from_spec(_losses_spec)
_losses_spec.loader.exec_module(_losses)
make_loss_fn = _losses.make_loss_fn
ic_values = _losses.ic_values
_compute_all = _losses._compute_all
apply_bc = _losses.apply_bc


# ── Config ──────────────────────────────────────────────────────────────────

from absl import app as _absl_app, flags as _absl_flags
from ml_collections import config_flags as _config_flags

_DEFAULT = os.path.join(os.path.dirname(__file__), 'configs', 'train_pinn.py')
_CONFIG = _config_flags.DEFINE_config_file('config', _DEFAULT, 'Config file')
FLAGS = _absl_flags.FLAGS


# ── Setup helpers ───────────────────────────────────────────────────────────

def _setup_dirs(config):
    log_dir = config.log.log_dir
    run = config.log.run_name or time.strftime('%Y%m%d-%H%M%S')
    run_dir = os.path.join(log_dir, run)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def _build_network(config, key):
    c = config.arch
    return make_split_pinn(
        spatial_dim=1,
        rank_p=c.rank_p,
        hidden_dim=c.hidden_dim,
        num_layers=c.num_layers,
        spatial_emb_dim=c.spatial_emb_dim,
        spatial_scale=c.spatial_scale,
        temporal_emb_dim=c.temporal_emb_dim,
        temporal_scale=c.temporal_scale,
        activation=c.activation,
        key=key,
    )


def _build_D(config, x_all):
    delta = config.nl.delta
    kernel = config.nl.kernel
    return build_D(x_all, delta, kernel=kernel)


def _build_optimizer(config):
    from pinn.optimizers import make_optimizer
    return make_optimizer(config)


def _nondim_params(config):
    phys = config.physical
    L_c = config.nondim.L_c
    T_c = config.nondim.T_c
    dc0 = phys.c_se - phys.c_le
    return (phys.D * T_c / L_c**2,          # N_CH
            2.0 * phys.A * phys.L * T_c,     # N_AC1
            phys.L * phys.omega_phi * T_c,   # N_AC2
            phys.L * phys.alpha_phi * T_c / L_c**2,  # N_AC3
            dc0, phys.c_le)


def _bc_info(x_all):
    """Build mask_free and bc_values for Dirichlet BC.

    Only endpoints are constrained — horizon truncation handles the rest
    (standard peridynamics for external boundaries).
    """
    N = len(x_all)
    mask_free = jnp.ones(N, dtype=bool)
    mask_free = mask_free.at[0].set(False)
    mask_free = mask_free.at[-1].set(False)
    bc_values = jnp.zeros(N)
    bc_values = bc_values.at[0].set(1.0)
    return mask_free, bc_values


# ── Training step ───────────────────────────────────────────────────────────

def _make_step(loss_fn, optim):
    @eqx.filter_jit
    def step(net, opt_state, t_batch, phi0, c0, w_ema):
        (loss, aux, grad, new_w) = loss_fn(net, t_batch, phi0, c0, w_ema)
        grad = jax.tree.map(jnp.nan_to_num, grad)
        updates, opt_state = optim.update(grad, opt_state, net)
        net = eqx.apply_updates(net, updates)
        return net, opt_state, loss, aux, new_w
    return step


# ── Evaluation ──────────────────────────────────────────────────────────────

def _eval_fem(net, fem_data, mask_free_eval, bc_values_eval, writer, step):
    x_fem = jnp.array(fem_data['mesh'])
    t_fem = jnp.array(fem_data['time'])
    phi_fem = jnp.array(fem_data['phi'])
    c_fem   = jnp.array(fem_data['c'])
    phi_pred, c_pred, _, _ = _compute_all(net, x_fem[:, None], t_fem)
    # Apply BC enforcement (same as training)
    phi_pred = apply_bc(phi_pred, mask_free_eval, bc_values_eval)
    c_pred   = apply_bc(c_pred,   mask_free_eval, bc_values_eval)
    # phi_pred: (N_x, N_t) → transpose to (N_t, N_x)
    phi_pred = phi_pred.T
    c_pred = c_pred.T

    rel_phi = float(jnp.sqrt(jnp.mean((phi_pred - phi_fem)**2))
                    / jnp.sqrt(jnp.mean(phi_fem**2)))
    rel_c   = float(jnp.sqrt(jnp.mean((c_pred - c_fem)**2))
                    / jnp.sqrt(jnp.mean(c_fem**2)))

    writer.add_scalar('eval/rel_L2_phi', rel_phi, step)
    writer.add_scalar('eval/rel_L2_c',   rel_c,   step)

    # Heatmaps: pred + error for φ and c (x→, t↓)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    err_phi = phi_pred - phi_fem
    err_c   = c_pred - c_fem
    cmap = 'coolwarm'
    extent = [float(x_fem[0]), float(x_fem[-1]), float(t_fem[-1]), float(t_fem[0])]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    im00 = axes[0, 0].imshow(phi_pred, cmap=cmap, vmin=0, vmax=1,
                              aspect='auto', origin='upper', extent=extent,
                              interpolation='bilinear', rasterized=True)
    axes[0, 0].set_title(r'$\phi$ (PINN)')
    plt.colorbar(im00, ax=axes[0, 0])

    im01 = axes[0, 1].imshow(err_phi, cmap=cmap, aspect='auto',
                              origin='upper', extent=extent,
                              interpolation='bilinear', rasterized=True)
    axes[0, 1].set_title(r'$\phi$ error')
    plt.colorbar(im01, ax=axes[0, 1])

    im10 = axes[1, 0].imshow(c_pred, cmap=cmap, vmin=0, vmax=1,
                              aspect='auto', origin='upper', extent=extent,
                              interpolation='bilinear', rasterized=True)
    axes[1, 0].set_title(r'$c$ (PINN)')
    plt.colorbar(im10, ax=axes[1, 0])

    im11 = axes[1, 1].imshow(err_c, cmap=cmap, aspect='auto',
                              origin='upper', extent=extent,
                              interpolation='bilinear', rasterized=True)
    axes[1, 1].set_title(r'$c$ error')
    plt.colorbar(im11, ax=axes[1, 1])

    for ax in axes.flat:
        ax.set_xlabel(r'$x^*$'); ax.set_ylabel(r'$t^*$')
    fig.suptitle(f'Step {step}  rel-L2 φ={rel_phi:.4f}  c={rel_c:.4f}', fontsize=11)
    fig.tight_layout()

    fpath = os.path.join(os.path.dirname(writer.logdir), f'eval_{step:06d}.png')
    fig.savefig(fpath, dpi=150)
    writer.add_figure('eval/heatmaps', fig, step)
    plt.close(fig)

    return rel_phi, rel_c


# ── Main ────────────────────────────────────────────────────────────────────

def _main(_):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    config = FLAGS.config
    rng = jr.PRNGKey(42)
    k_net, k_samp = jr.split(rng)

    # --- dirs & logging ---
    run_dir = _setup_dirs(config)
    writer = SummaryWriter(run_dir)
    log.info("Logging to %s", run_dir)

    # --- spatial grid & BC ---
    x_all = jnp.linspace(0.0, 1.0, config.sample.Nx)[:, None]
    delta = config.nl.delta
    mask_free, bc_values = _bc_info(x_all)
    n_free = int(jnp.sum(mask_free))
    log.info("Spatial: %d points, delta=%.2f, free=%d, constrained=%d",
             len(x_all), delta, n_free, len(x_all) - n_free)

    net = _build_network(config, k_net)
    D_full = _build_D(config, x_all)
    D = DWithBC(D_full, mask_free, bc_values)  # BC enforced inside D @ u
    nd_params = _nondim_params(config)
    lfn = make_loss_fn(x_all, D, mask_free, bc_values, nd_params,
                       alpha=config.train.alpha_gradnorm,
                       ac_scale=config.loss.ac_scale,
                       ch_scale=config.loss.ch_scale,
                       gradnorm_cap=config.loss.gradnorm_weight_cap)
    optim = _build_optimizer(config)
    opt_state = optim.init(eqx.filter(net, eqx.is_array))
    step_fn = _make_step(lfn, optim)

    # --- initial condition ---
    phi0, c0 = ic_values(x_all, config.physical, config.nondim.L_c)

    # --- FEM data for eval ---
    fem_data = np.load(config.fem.data_path)
    N_fem = len(fem_data['mesh'])
    mask_free_eval = jnp.ones(N_fem, bool)
    mask_free_eval = mask_free_eval.at[0].set(False)
    mask_free_eval = mask_free_eval.at[-1].set(False)
    bc_values_eval = jnp.zeros(N_fem)
    bc_values_eval = bc_values_eval.at[0].set(1.0)

    # --- training loop ---
    w_ema = jnp.array([1.0, 1.0, 1.0, 1.0])
    t_start = time.perf_counter()

    for it in range(1, config.train.iterations + 1):
        key_t = jr.fold_in(k_samp, it)
        t_batch = jr.uniform(key_t, (config.sample.Nt,), minval=0.0, maxval=1.0)

        net, opt_state, loss_val, aux, w_ema = step_fn(
            net, opt_state, t_batch, phi0, c0, w_ema)

        if it % config.train.log_every == 0:
            elapsed = time.perf_counter() - t_start
            writer.add_scalar('loss/total', float(loss_val), it)
            for k, v in aux.items():
                writer.add_scalar(f'loss/{k}', float(v), it)
            lr = config.train.lr_init * config.train.lr_decay_rate**(it // config.train.lr_decay_every)
            writer.add_scalar('train/lr', float(max(lr, config.train.lr_min)), it)
            log.info("[%5d] loss=%.3e  ac=%.3e ch=%.3e bc=%.3e ic=%.3e  "
                     "w=[%.1f,%.1f,%.1f,%.0f]  (%.1fs)",
                     it, float(loss_val), float(aux['loss_ac']),
                     float(aux['loss_ch']), float(aux['loss_bc']),
                     float(aux['loss_ic']),
                     float(aux['weight_ac']), float(aux['weight_ch']),
                     float(aux['weight_bc']), float(aux['weight_ic']),
                     elapsed)

        if it % config.train.eval_every == 0:
            rel_phi, rel_c = _eval_fem(net, fem_data, mask_free_eval, bc_values_eval, writer, it)
            log.info("--- eval @ %d: rel-L2 phi=%.4f  c=%.4f ---", it, rel_phi, rel_c)

    writer.close()
    log.info("Done. Logs at %s", run_dir)


if __name__ == '__main__':
    _absl_app.run(_main)
