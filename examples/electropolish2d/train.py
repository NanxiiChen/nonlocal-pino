"""Training script for PINN on 2D KKS electropolishing — rough interface.

Usage (from project root):
  python examples/electropolish2d/train.py
  python examples/electropolish2d/train.py --config=examples/electropolish2d/configs/train_base.py
  python examples/electropolish2d/train.py --config.nl.method=fd --config.log.run_name=fd_run
"""

import importlib.util as _iu
import logging, os, time

import jax, jax.numpy as jnp, jax.random as jr

jax.config.update("jax_default_matmul_precision", "highest")
import equinox as eqx
import numpy as np
from scipy.stats.qmc import LatinHypercube
from tensorboardX import SummaryWriter

log = logging.getLogger(__name__)

from pinn.operators import build_D
from pinn.arch import make_pinn, make_film_pinn


def _load_losses():
    path = os.path.join(os.path.dirname(__file__), 'losses.py')
    spec = _iu.spec_from_file_location('electropolish2d_losses', path)
    mod = _iu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── Config ──────────────────────────────────────────────────────────────────

from absl import app as _absl_app, flags as _absl_flags
from ml_collections import config_flags as _config_flags

_DEFAULT = os.path.join(os.path.dirname(__file__), 'configs', 'train_base.py')
_CONFIG = _config_flags.DEFINE_config_file('config', _DEFAULT, 'Config file')
FLAGS = _absl_flags.FLAGS


# ── Setup helpers ───────────────────────────────────────────────────────────

def _setup_dirs(config):
    log_dir = config.log.log_dir
    run = config.log.run_name or time.strftime('%Y%m%d-%H%M%S')
    run_dir = os.path.join(log_dir, run)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def _get_t_weights(config, t_batch, step):
    cfg = config.loss
    if getattr(cfg, 't_ramp_steps', 0) <= 0:
        return None
    if step > cfg.t_ramp_steps:
        return None
    t_norm = (t_batch - config.sample.t_min) / (1.0 - config.sample.t_min)
    t_norm = jnp.clip(t_norm, 0.0, 1.0)
    return jnp.array(cfg.t_ramp_early + (cfg.t_ramp_late - cfg.t_ramp_early) * t_norm)


def _build_network(config, key):
    c = config.arch
    hc = None
    if c.use_hard_constraint:
        from pinn.arch.hard_constraint import HardConstraint
        phys = config.physical
        hc = HardConstraint(phys.c_se, phys.c_le)
    if c.type == "film":
        return make_film_pinn(spatial_dim=2, out_dim=2, hidden_dim=c.hidden_dim,
                              num_layers=c.num_layers,
                              film_hidden_dim=c.film_hidden_dim,
                              film_num_layers=c.film_num_layers,
                              activation=c.activation,
                              spatial_emb_dim=c.spatial_emb_dim,
                              spatial_scale=c.spatial_scale,
                              temporal_emb_dim=c.temporal_emb_dim,
                              temporal_scale=c.temporal_scale,
                              use_spatial_emb=c.use_spatial_emb,
                              use_temporal_emb=c.use_temporal_emb,
                              hard_constraint=hc, key=key)
    return make_pinn(spatial_dim=2, out_dim=2, hidden_dim=c.hidden_dim,
                     num_layers=c.num_layers, trunk_type=c.type,
                     activation=c.activation,
                     spatial_emb_dim=c.spatial_emb_dim,
                     spatial_scale=c.spatial_scale,
                     temporal_emb_dim=c.temporal_emb_dim,
                     temporal_scale=c.temporal_scale,
                     use_spatial_emb=c.use_spatial_emb,
                     use_temporal_emb=c.use_temporal_emb,
                     tensor_split=c.arch.tensor_split,
                     hard_constraint=hc, dirichlet_bc=None, key=key)


def _build_D(config, x_all):
    if config.nl.method == "local":
        return None   # autograd Laplacian, no pre-built operator
    if config.nl.method == "fd":
        from pinn.fd import FiniteDiff
        return FiniteDiff(x_all, Nx=config.sample.Nx, Ny=config.sample.Ny)
    return build_D(x_all, config.nl.delta, kernel=config.nl.kernel,
                   method=config.nl.method)


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


def _bc_info(config):
    """Sample BC points.

    Top (y=0.5): liquid — φ=0, c=0.
    Bottom (y=0): solid — φ=1, c=1.
    Left/Right: Neumann — no BC points (key ablation point!).
    """
    rng = np.random.RandomState(42)

    # Top edge: y=0.5, x ∈ [0, 1]
    xs_top = rng.uniform(0.0, 1.0, config.sample.N_bc_top)
    x_bc_top = jnp.column_stack([xs_top, np.full_like(xs_top, 0.5)])

    # Bottom edge: y=0, x ∈ [0, 1]
    xs_bot = rng.uniform(0.0, 1.0, config.sample.N_bc_bottom)
    x_bc_bottom = jnp.column_stack([xs_bot, np.zeros_like(xs_bot)])

    return jnp.array(x_bc_top), jnp.array(x_bc_bottom)


def _sample_ic_points(config):
    """Sample IC points: global uniform + local oversampling near interface."""
    rng = np.random.RandomState(42)
    N_global = config.sample.N_ic_global
    N_local = config.sample.N_ic_local

    # Global uniform samples
    xs_g = rng.uniform(0.0, 1.0, N_global)
    ys_g = rng.uniform(0.0, 0.5, N_global)
    points = [np.column_stack([xs_g, ys_g])]

    # Local oversampling near rough interface: y ∈ [y0-2*A, y0+2*A]
    y_lo = max(config.surface.y0 - 2.0 * config.surface.amplitude, 0.0)
    y_hi = min(config.surface.y0 + 2.0 * config.surface.amplitude, 0.5)
    xs_l = rng.uniform(0.0, 1.0, N_local)
    ys_l = rng.uniform(y_lo, y_hi, N_local)
    points.append(np.column_stack([xs_l, ys_l]))

    return jnp.array(np.vstack(points)[:N_global + N_local])


# ── Training step ───────────────────────────────────────────────────────────

def _make_step(loss_fn, optim):
    @eqx.filter_jit
    def step(net, opt_state, t_batch, phi0, c0, eps, train_ac, train_ch, w_ema,
             bias, update_w, iface_gamma, t_weights=None,
             x_points=None, sub_data=None, core_mask=None):
        (loss, aux, grad, new_w) = loss_fn(
            net, t_batch, phi0, c0, eps, train_ac, train_ch, w_ema, bias,
            update_w, iface_gamma, t_weights,
            x_points=x_points, sub_data=sub_data, core_mask=core_mask)
        grad = jax.tree.map(jnp.nan_to_num, grad)
        updates, opt_state = optim.update(grad, opt_state, net)
        net = eqx.apply_updates(net, updates)
        return net, opt_state, loss, aux, new_w
    return step


# ── Evaluation ──────────────────────────────────────────────────────────────

def _eval_fem(net, fem_data, writer, step):
    X = jnp.array(fem_data['X'])
    Y = jnp.array(fem_data['Y'])
    t_fem = jnp.array(fem_data['time'])
    phi_fem = jnp.array(fem_data['phi'])
    c_fem = jnp.array(fem_data['c'])

    x_fem = jnp.stack([X.ravel(), Y.ravel()], axis=-1)
    shape_fem = phi_fem.shape

    out = jax.vmap(lambda t: jax.vmap(net, in_axes=(0, None))(x_fem, t))(t_fem)
    phi_pred = out[..., 0].reshape(shape_fem)
    c_pred = out[..., 1].reshape(shape_fem)

    rel_phi = float(jnp.sqrt(jnp.mean((phi_pred - phi_fem)**2))
                    / jnp.sqrt(jnp.mean(phi_fem**2)))
    rel_c   = float(jnp.sqrt(jnp.mean((c_pred - c_fem)**2))
                    / jnp.sqrt(jnp.mean(c_fem**2)))

    writer.add_scalar('eval/rel_L2_phi', rel_phi, step)
    writer.add_scalar('eval/rel_L2_c',   rel_c,   step)

    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    target_times = [0.0, 0.25, 0.5, 0.75, 1.0]
    indices = [int(np.argmin(np.abs(np.array(t_fem) - tt))) for tt in target_times]
    indices = sorted(set(indices))
    n = len(indices)
    err_phi = jnp.abs(phi_pred - phi_fem)
    max_err = float(jnp.max(err_phi[jnp.array(indices)]))

    fig, axes = plt.subplots(2, n, figsize=(3*n, 5), squeeze=False)
    for j, i in enumerate(indices):
        axes[0][j].imshow(phi_pred[i], extent=[0, 1, 0, 0.5],
                          origin='lower', cmap='coolwarm',
                          vmin=0, vmax=1, interpolation='bilinear')
        axes[0][j].set_title(rf'$\phi$ PINN  t*={t_fem[i]:.2f}')
        axes[0][j].set_aspect('equal')

        im1 = axes[1][j].imshow(err_phi[i], extent=[0, 1, 0, 0.5],
                                 origin='lower', cmap='coolwarm',
                                 vmin=0, vmax=max_err, interpolation='bilinear')
        axes[1][j].set_title(rf'$|\phi - \phi_{{FEM}}|$  t*={t_fem[i]:.2f}')
        axes[1][j].set_aspect('equal')

    fig.tight_layout(rect=[0, 0.1, 1, 1])
    cbar_ax = fig.add_axes([0.15, 0.1, 0.7, 0.025])
    fig.colorbar(im1, cax=cbar_ax, orientation='horizontal', label='|error|')
    writer.add_figure('eval/snapshots', fig, step)
    plt.close(fig)

    return rel_phi, rel_c


# ── Main ────────────────────────────────────────────────────────────────────

def _main(_):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    config = FLAGS.config
    # Default "highest" = exact FP32 (unchanged behavior); opt-in override only.
    jax.config.update("jax_default_matmul_precision",
                      config.train.get('matmul_precision', 'highest'))
    rng = jr.PRNGKey(42)
    k_net, k_samp = jr.split(rng)

    loss_mod = _load_losses()
    make_loss_fn = loss_mod.make_loss_fn
    ic_values = loss_mod.ic_values

    # --- dirs & logging ---
    run_dir = _setup_dirs(config)
    writer = SummaryWriter(run_dir)
    log.info("Logging to %s", run_dir)
    with open(os.path.join(run_dir, 'config.json'), 'w') as f:
        f.write(config.to_json_best_effort())

    # --- spatial grid (2D structured) ---
    xs = jnp.linspace(0.0, 1.0, config.sample.Nx)
    ys = jnp.linspace(0.0, 0.5, config.sample.Ny)
    Xv, Yv = jnp.meshgrid(xs, ys, indexing='xy')
    x_base = jnp.stack([Xv.ravel(), Yv.ravel()], axis=-1)

    x_bc_top, x_bc_bottom = _bc_info(config)
    log.info("PDE: %d×%d=%d points, delta=%.2f, bc_top=%d, bc_bottom=%d",
             config.sample.Nx, config.sample.Ny, len(x_base), config.nl.delta,
             len(x_bc_top), len(x_bc_bottom))

    net = _build_network(config, k_net)
    D = _build_D(config, x_base)
    nd_params = _nondim_params(config)

    causal = None
    causal_eps_ac = jnp.array(1.0)
    causal_eps_ch = jnp.array(1.0)
    if config.loss.use_causal:
        from pinn.causal import CausalWeightor
        n_ch = config.causal.n_chunks if config.causal.n_chunks > 0 else None
        causal = CausalWeightor(n_chunks=n_ch)
        causal_eps_ac = jnp.array(config.causal.eps_init)
        causal_eps_ch = jnp.array(config.causal.eps_init)

    # --- IC ---
    x_ic = _sample_ic_points(config)
    log.info("IC points: %d (global=%d + local=%d)",
             len(x_ic), config.sample.N_ic_global, config.sample.N_ic_local)
    phi0, c0 = ic_values(x_ic, config.physical, config.nondim.L_c, config.surface)
    log.info("IC: phi min=%.3f max=%.3f  c min=%.3f max=%.3f",
             float(jnp.min(phi0)), float(jnp.max(phi0)),
             float(jnp.min(c0)), float(jnp.max(c0)))

    lfn = make_loss_fn(x_base, D, x_bc_top, x_bc_bottom, nd_params,
                       x_ic=x_ic,
                       alpha=config.train.alpha_gradnorm,
                       ac_scale=config.loss.ac_scale,
                       ch_scale=config.loss.ch_scale,
                       gradnorm_cap=config.loss.gradnorm_weight_cap,
                       causal=causal)
    optim = _build_optimizer(config)
    opt_state = optim.init(eqx.filter(net, eqx.is_array))
    step_fn = _make_step(lfn, optim)

    # --- bias schedule ---
    schedule = list(config.loss.bias_schedule)
    default_bias = jnp.array(config.loss.gradnorm_bias, dtype=jnp.float32)
    n_bias = len(default_bias)
    def get_bias(it):
        for entry in schedule:
            if len(entry) < n_bias + 1:
                continue
            end_step = entry[0]
            if it <= end_step:
                return jnp.array(entry[1:n_bias + 1], dtype=jnp.float32)
        return default_bias

    # --- FEM data for eval ---
    fem_data = np.load(config.fem.data_path)

    # --- pre-build spatial sub-graph pool ---
    subgraph_pool = None
    _K_core = config.sample.get('K_core', 0)
    if _K_core > 0 and _K_core < len(x_base):
        if hasattr(D, 'build_subgraph_pool'):
            n_pool = config.sample.get('K_pool', 200)
            log.info("Pre-building sub-graph pool (%d entries, K_core=%d) ...",
                     n_pool, _K_core)
            _t0 = time.perf_counter()
            raw_pool = D.build_subgraph_pool(_K_core, n_pool)
            _K_alls = [r[3] for r in raw_pool]
            log.info("  K_all: min=%d max=%d avg=%d (K_core=%d, N=%d)",
                     min(_K_alls), max(_K_alls), int(np.mean(_K_alls)),
                     _K_core, D._N)
            subgraph_pool = [
                (x_base[r[0]], r[1], r[2]) for r in raw_pool
            ]
            del raw_pool
            _N_pad = len(subgraph_pool[0][0])
            _E_pad = len(subgraph_pool[0][1][0])
            log.info("Pool ready in %.1fs: N_pad=%d E_pad=%d",
                     time.perf_counter() - _t0, _N_pad, _E_pad)

    # --- LHS sampler (local mode only) ---
    lhs_sampler = None
    N_lhs = 0
    if config.nl.method == "local":
        N_lhs = config.sample.get('N_lhs', _K_core)
        lhs_sampler = LatinHypercube(d=2, seed=42)  # (x, y)
        log.info("LHS sampler: N_lhs=%d (resampled each step)", N_lhs)

    # --- training loop ---
    w_ema = jnp.ones(5)
    t_start = time.perf_counter()

    for it in range(config.train.iterations):
        key_t = jr.fold_in(k_samp, it)
        t_batch = jr.uniform(key_t, (config.sample.Nt,),
                             minval=config.sample.t_min, maxval=1.0)

        eps = jnp.minimum(causal_eps_ac, causal_eps_ch)
        stagger_steps = config.loss.stagger_steps
        if it < stagger_steps:
            phase = (it // config.loss.stagger_switch_every) % 2
            train_ac = jnp.array(phase == 0)
            train_ch = jnp.array(phase == 1)
        else:
            train_ac = jnp.array(True)
            train_ch = jnp.array(True)
        update_w = (it % config.train.gradnorm_update_every == 0)

        gamma_decay = config.loss.iface_gamma_decay_steps
        if gamma_decay > 0 and it <= gamma_decay:
            iface_gamma = jnp.array(config.loss.iface_gamma_init
                                    * (1.0 - it / gamma_decay))
        else:
            iface_gamma = jnp.array(config.loss.iface_gamma_decay_to)

        t_weights = _get_t_weights(config, t_batch, it)

        if subgraph_pool is not None:
            _pi = it % len(subgraph_pool)
            x_points, sub_data, core_mask = subgraph_pool[_pi]
        elif lhs_sampler is not None:
            x_points = jnp.array(lhs_sampler.random(n=N_lhs), dtype=jnp.float32)
            sub_data = core_mask = None
        else:
            x_points = x_base
            sub_data = core_mask = None

        net, opt_state, loss_val, aux, w_ema = step_fn(
            net, opt_state, t_batch, phi0, c0, eps, train_ac, train_ch, w_ema,
            get_bias(it), update_w, iface_gamma, t_weights,
            x_points=x_points, sub_data=sub_data, core_mask=core_mask)

        if it % config.log.log_every == 0:
            elapsed = time.perf_counter() - t_start
            writer.add_scalar('loss/total', float(loss_val), it)
            for k, v in aux.items():
                if v is not None and (not isinstance(v, jnp.ndarray) or v.ndim == 0):
                    writer.add_scalar(f'loss/{k}', float(v), it)
            lr = config.train.lr_init * config.train.lr_decay_rate**(it // config.train.lr_decay_every)
            writer.add_scalar('train/lr', float(max(lr, config.train.lr_min)), it)
            log.info("[%5d] loss=%.3e  ac=%.3e ch=%.3e bc_t=%.3e bc_b=%.3e ic=%.3e  "
                     "w=[%.1e,%.1e,%.1e,%.1e,%.1e]  (%.1fs)",
                     it, float(loss_val), float(aux['loss_ac']),
                     float(aux['loss_ch']),
                     float(aux['loss_bc_top']), float(aux['loss_bc_bot']),
                     float(aux['loss_ic']),
                     float(aux['weight_ac']), float(aux['weight_ch']),
                     float(aux['weight_bc_top']), float(aux['weight_bc_bot']),
                     float(aux['weight_ic']), elapsed)

        if causal is not None and aux.get('causal_w_ac') is not None:
            cfg = config.causal
            causal_eps_ac = causal.update_eps(
                aux['causal_w_ac'], causal_eps_ac,
                cfg.step_size, cfg.max_eps, cfg.max_last_weight, cfg.min_mean_weight)
            causal_eps_ch = causal.update_eps(
                aux['causal_w_ch'], causal_eps_ch,
                cfg.step_size, cfg.max_eps, cfg.max_last_weight, cfg.min_mean_weight)
            if it % config.log.log_every == 0:
                writer.add_scalar('causal/eps_ac', float(causal_eps_ac), it)
                writer.add_scalar('causal/eps_ch', float(causal_eps_ch), it)

        if it % config.log.eval_every == 0:
            rel_phi, rel_c = _eval_fem(net, fem_data, writer, it)
            log.info("--- eval @ %d: rel-L2 phi=%.4f  c=%.4f ---", it, rel_phi, rel_c)

            if config.log.save_ckpt:
                ckpt_path = os.path.join(run_dir, f'ckpt_{it:06d}.eqx')
                eqx.tree_serialise_leaves(ckpt_path, net)
                log.info("  Saved checkpoint: %s", ckpt_path)

    writer.close()
    log.info("Done. Logs at %s", run_dir)


if __name__ == '__main__':
    _absl_app.run(_main)
