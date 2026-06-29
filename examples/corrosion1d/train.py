"""Training script for PINN on 1D KKS phase-field corrosion.

Supports full MLP, modified MLP, split, and FiLM architectures via config.arch.type.

Usage (from project root):
  python examples/corrosion1d/train.py
  python examples/corrosion1d/train.py --config=examples/corrosion1d/configs/train_diffusion.py
  python examples/corrosion1d/train.py --config.train.iterations=1000 --config.nl.delta=0.05
"""

import logging, os, time

import jax, jax.numpy as jnp, jax.random as jr

jax.config.update("jax_default_matmul_precision", "highest")
import equinox as eqx
import numpy as np
from tensorboardX import SummaryWriter

log = logging.getLogger(__name__)

from pinn.operators import build_D
from pinn.arch import make_pinn, make_split_pinn, make_film_pinn

import importlib.util as _iu
_loss_mod = None

def _load_losses():
    global _loss_mod
    if _loss_mod is not None:
        return _loss_mod
    path = os.path.join(os.path.dirname(__file__), 'losses.py')
    spec = _iu.spec_from_file_location('corrosion1d_losses', path)
    _loss_mod = _iu.module_from_spec(spec)
    spec.loader.exec_module(_loss_mod)
    return _loss_mod

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

    if c.type == "split":
        return make_split_pinn(
            spatial_dim=1, rank_p=c.rank_p, hidden_dim=c.hidden_dim,
            num_layers=c.num_layers, spatial_emb_dim=c.spatial_emb_dim,
            spatial_scale=c.spatial_scale, temporal_emb_dim=c.temporal_emb_dim,
            temporal_scale=c.temporal_scale, activation=c.activation,
            use_bias=c.use_bias, use_spatial_emb=c.use_spatial_emb,
            use_temporal_emb=c.use_temporal_emb, key=key)
    elif c.type in ("mlp", "modified_mlp"):
        hc = None
        if c.use_hard_constraint:
            from pinn.arch.hard_constraint import HardConstraint
            phys = config.physical
            hc = HardConstraint(phys.c_se, phys.c_le)
        return make_pinn(spatial_dim=1, out_dim=2, hidden_dim=c.hidden_dim,
                         num_layers=c.num_layers, trunk_type=c.type,
                         activation=c.activation,
                         spatial_emb_dim=c.spatial_emb_dim,
                         spatial_scale=c.spatial_scale,
                         temporal_emb_dim=c.temporal_emb_dim,
                         temporal_scale=c.temporal_scale,
                         use_fourier=c.use_spatial_emb or c.use_temporal_emb,
                         hard_constraint=hc, key=key)
    elif c.type == "film":
        hc = None
        if c.use_hard_constraint:
            from pinn.arch.hard_constraint import HardConstraint
            phys = config.physical
            hc = HardConstraint(phys.c_se, phys.c_le)
        return make_film_pinn(spatial_dim=1, out_dim=2, hidden_dim=c.hidden_dim,
                              num_layers=c.num_layers,
                              film_hidden_dim=c.film_hidden_dim,
                              activation=c.activation,
                              spatial_emb_dim=c.spatial_emb_dim,
                              spatial_scale=c.spatial_scale,
                              temporal_emb_dim=c.temporal_emb_dim,
                              temporal_scale=c.temporal_scale,
                              use_fourier=c.use_spatial_emb or c.use_temporal_emb,
                              hard_constraint=hc, key=key)
    else:
        raise ValueError(f"Unknown arch.type: {c.type}")


def _build_D(config, x_all):
    delta = config.nl.delta
    kernel = config.nl.kernel
    method = config.nl.method
    return build_D(x_all, delta, kernel=kernel, method=method)


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
    N = len(x_all)
    mask_free = jnp.ones(N, dtype=bool)
    mask_free = mask_free.at[0].set(False)
    mask_free = mask_free.at[-1].set(False)
    bc_values = jnp.zeros(N)
    bc_values = bc_values.at[0].set(1.0)
    bc_values = bc_values.at[-1].set(0.0)
    return mask_free, bc_values


# ── Training step ───────────────────────────────────────────────────────────

def _make_step(loss_fn, optim):
    @eqx.filter_jit
    def step(net, opt_state, t_batch, phi0, c0, eps, train_ac, train_ch, w_ema, bias):
        (loss, aux, grad, new_w) = loss_fn(net, t_batch, phi0, c0, eps, train_ac, train_ch, w_ema, bias)
        grad = jax.tree.map(jnp.nan_to_num, grad)
        updates, opt_state = optim.update(grad, opt_state, net)
        net = eqx.apply_updates(net, updates)
        return net, opt_state, loss, aux, new_w
    return step


# ── Evaluation ──────────────────────────────────────────────────────────────

def _eval_fem(net, fem_data, compute_all, writer, step):
    x_fem = jnp.array(fem_data['mesh'])
    t_fem = jnp.array(fem_data['time'])
    phi_fem = jnp.array(fem_data['phi'])
    c_fem   = jnp.array(fem_data['c'])
    phi_pred, c_pred, _, _ = compute_all(net, x_fem[:, None], t_fem)
    phi_pred = phi_pred.T
    c_pred = c_pred.T

    rel_phi = float(jnp.sqrt(jnp.mean((phi_pred - phi_fem)**2))
                    / jnp.sqrt(jnp.mean(phi_fem**2)))
    rel_c   = float(jnp.sqrt(jnp.mean((c_pred - c_fem)**2))
                    / jnp.sqrt(jnp.mean(c_fem**2)))

    writer.add_scalar('eval/rel_L2_phi', rel_phi, step)
    writer.add_scalar('eval/rel_L2_c',   rel_c,   step)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    err_phi = jnp.abs(phi_pred - phi_fem)
    err_c   = jnp.abs(c_pred - c_fem)
    extent = [float(x_fem[0]), float(x_fem[-1]), float(t_fem[0]), float(t_fem[-1])]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    cmap_pred = 'coolwarm'
    im00 = axes[0, 0].imshow(phi_pred, cmap=cmap_pred, vmin=0, vmax=1,
                              aspect='auto', origin='lower', extent=extent,
                              interpolation='bilinear', rasterized=True)
    axes[0, 0].set_title(r'$\phi$ (PINN)')
    plt.colorbar(im00, ax=axes[0, 0])

    im01 = axes[0, 1].imshow(err_phi, cmap='coolwarm', aspect='auto',
                              origin='lower', extent=extent,
                              interpolation='bilinear', rasterized=True)
    axes[0, 1].set_title(r'$\phi$ error')
    plt.colorbar(im01, ax=axes[0, 1])

    im10 = axes[1, 0].imshow(c_pred, cmap=cmap_pred, vmin=0, vmax=1,
                              aspect='auto', origin='lower', extent=extent,
                              interpolation='bilinear', rasterized=True)
    axes[1, 0].set_title(r'$c$ (PINN)')
    plt.colorbar(im10, ax=axes[1, 0])

    im11 = axes[1, 1].imshow(err_c, cmap='coolwarm', aspect='auto',
                              origin='lower', extent=extent,
                              interpolation='bilinear', rasterized=True)
    axes[1, 1].set_title(r'$c$ error')
    plt.colorbar(im11, ax=axes[1, 1])

    for ax in axes.flat:
        ax.set_xlabel(r'$x^*$'); ax.set_ylabel(r'$t^*$')
    fig.suptitle(f'Step {step}  rel-L2 φ={rel_phi:.4f}  c={rel_c:.4f}', fontsize=11)
    fig.tight_layout()

    writer.add_figure('eval/heatmaps', fig, step)
    plt.close(fig)

    return rel_phi, rel_c


def _diag_residual(net, x_fem, t_fem, delta, kernel, nd_params, compute_all, writer, step):
    def _h(phi): return -2.0 * phi**3 + 3.0 * phi**2
    def _hp(phi): return -6.0 * phi**2 + 6.0 * phi
    def _gp(phi): return 4.0 * phi**3 - 6.0 * phi**2 + 2.0 * phi

    N_CH, N_AC1, N_AC2, N_AC3, dc0, c_le = nd_params
    D_fem = build_D(x_fem[:, None], delta, kernel=kernel)

    phi, c, dphi_dt, _ = compute_all(net, x_fem[:, None], t_fem)
    phi, c, dphi_dt = phi.T, c.T, dphi_dt.T

    nl_phi = (D_fem @ phi.T).T
    r_t = dphi_dt
    r_D = N_AC3 * nl_phi
    r_g = N_AC2 * _gp(phi)
    r_drv = -N_AC1 * (c - _h(phi) * dc0 - c_le) * dc0 * _hp(phi)
    r_ac = r_t + r_D + r_g + r_drv

    for name, val in [('ac/dphi_dt', r_t), ('ac/N_AC3_Dphi', r_D),
                       ('ac/N_AC2_gp', r_g), ('ac/driving', r_drv),
                       ('ac/total', r_ac)]:
        rms = float(jnp.sqrt(jnp.mean(val**2)))
        writer.add_scalar(f'diag/{name}_rms', rms, step)

    mu = c - _h(phi) * dc0 - c_le
    nl_mu = (D_fem @ mu.T).T
    r_ch_t = compute_all(net, x_fem[:, None], t_fem)[3].T
    r_ch = r_ch_t + N_CH * nl_mu
    writer.add_scalar('diag/ch_total_rms', float(jnp.sqrt(jnp.mean(r_ch**2))), step)

    log.info("  AC diag: dt=%.2e  Dphi=%.2e  gp=%.2e  drv=%.2e  → ac_res=%.2e",
             float(jnp.sqrt(jnp.mean(r_t**2))),
             float(jnp.sqrt(jnp.mean(r_D**2))),
             float(jnp.sqrt(jnp.mean(r_g**2))),
             float(jnp.sqrt(jnp.mean(r_drv**2))),
             float(jnp.sqrt(jnp.mean(r_ac**2))))


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
    with open(os.path.join(run_dir, 'config.json'), 'w') as f:
        f.write(config.to_json_best_effort())

    # --- spatial grid & BC ---
    x_all = jnp.linspace(0.0, 1.0, config.sample.Nx)[:, None]
    delta = config.nl.delta
    mask_free, bc_values = _bc_info(x_all)
    n_free = int(jnp.sum(mask_free))
    log.info("Spatial: %d points, delta=%.2f, free=%d, constrained=%d",
             len(x_all), delta, n_free, len(x_all) - n_free)

    # --- architecture-specific compute functions ---
    lm = _load_losses()
    make_loss_fn = lm.make_loss_fn
    ic_values = lm.ic_values
    arch_type = config.arch.type
    if arch_type == "split":
        compute_all, eval_t0 = lm._make_split_compute()
    elif arch_type in ("mlp", "modified_mlp"):
        compute_all, eval_t0 = lm._make_full_compute()
    else:
        raise ValueError(f"Unknown arch.type: {arch_type}")

    net = _build_network(config, k_net)
    D_full = _build_D(config, x_all)
    D = D_full
    nd_params = _nondim_params(config)
    causal = None
    causal_eps_ac = jnp.array(1.0)
    causal_eps_ch = jnp.array(1.0)
    if config.loss.use_causal:
        from pinn.causal import CausalWeightor
        causal = CausalWeightor(n_chunks=config.causal.n_chunks)
        causal_eps_ac = jnp.array(config.causal.eps_init)
        causal_eps_ch = jnp.array(config.causal.eps_init)
    lfn = make_loss_fn(x_all, D, mask_free, bc_values, nd_params,
                       compute_all=compute_all, eval_t0=eval_t0,
                       alpha=config.train.alpha_gradnorm,
                       ac_scale=config.loss.ac_scale,
                       ch_scale=config.loss.ch_scale,
                       gradnorm_cap=config.loss.gradnorm_weight_cap,
                       causal=causal,
                       iface_gamma=config.loss.iface_gamma)
    optim = _build_optimizer(config)
    opt_state = optim.init(eqx.filter(net, eqx.is_array))
    step_fn = _make_step(lfn, optim)

    # --- initial condition ---
    phi0, c0 = ic_values(x_all, config.physical, config.nondim.L_c, center=config.init.center)

    # --- bias schedule ---
    schedule = list(config.loss.bias_schedule)
    default_bias = jnp.array(config.loss.gradnorm_bias, dtype=jnp.float32)
    n_bias = len(default_bias)

    def get_bias(it):
        for entry in schedule:
            if not isinstance(entry, (tuple, list)) or len(entry) < n_bias + 1:
                continue
            end_step = entry[0]
            if it <= end_step:
                return jnp.array(entry[1:n_bias + 1], dtype=jnp.float32)
        return default_bias

    # --- FEM data for eval ---
    fem_data = np.load(config.fem.data_path)

    # --- training loop ---
    w_ema = jnp.array([1.0, 1.0, 1.0, 1.0])
    t_start = time.perf_counter()

    for it in range(1, config.train.iterations + 1):
        key_t = jr.fold_in(k_samp, it)
        t_batch = jr.uniform(key_t, (config.sample.Nt,), minval=config.sample.t_min, maxval=1.0)

        eps = jnp.minimum(causal_eps_ac, causal_eps_ch)
        if config.loss.use_stagger:
            phase = (it // config.loss.stagger_switch_every) % 2
            train_ac = jnp.array(phase == 0)
            train_ch = jnp.array(phase == 1)
        else:
            train_ac = jnp.array(True)
            train_ch = jnp.array(True)
        net, opt_state, loss_val, aux, w_ema = step_fn(
            net, opt_state, t_batch, phi0, c0, eps, train_ac, train_ch, w_ema,
            get_bias(it))

        if it % config.log.log_every == 0:
            elapsed = time.perf_counter() - t_start
            writer.add_scalar('loss/total', float(loss_val), it)
            for k, v in aux.items():
                if v is not None and (not isinstance(v, jnp.ndarray) or v.ndim == 0):
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
            rel_phi, rel_c = _eval_fem(net, fem_data, compute_all, writer, it)
            log.info("--- eval @ %d: rel-L2 phi=%.4f  c=%.4f ---", it, rel_phi, rel_c)

            if config.log.run_diag:
                x_fem = jnp.array(fem_data['mesh'])
                t_fem = jnp.array(fem_data['time'])
                _diag_residual(net, x_fem, t_fem, delta, config.nl.kernel,
                               nd_params, compute_all, writer, it)

            if config.log.save_ckpt:
                ckpt_path = os.path.join(run_dir, f'ckpt_{it:06d}.eqx')
                eqx.tree_serialise_leaves(ckpt_path, net)
                log.info("  Saved checkpoint: %s", ckpt_path)

            if causal is not None and aux.get('causal_w_ac') is not None:
                import matplotlib; matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                bins_mid = (causal.bins[1:] + causal.bins[:-1]) / 2
                fig, axes = plt.subplots(1, 2, figsize=(10, 3))
                for _i, (w, chunks, label) in enumerate([
                    (aux['causal_w_ac'], aux['causal_chunks_ac'], 'AC'),
                    (aux['causal_w_ch'], aux['causal_chunks_ch'], 'CH')]):
                    axes[0].plot(bins_mid, w, 'o-', label=label)
                    axes[1].plot(bins_mid, chunks, 'o-', label=label)
                axes[0].set_title(f'Causal weights (eps_ac={float(causal_eps_ac):.3f}, ch={float(causal_eps_ch):.3f})')
                axes[0].set_xlabel('t chunk'); axes[0].set_ylabel('weight'); axes[0].legend()
                axes[1].set_title('Loss per chunk'); axes[1].set_xlabel('t chunk'); axes[1].set_ylabel('MAE')
                axes[1].set_yscale('log'); axes[1].legend()
                fig.tight_layout()
                writer.add_figure('causal/info', fig, it)
                plt.close(fig)

    writer.close()
    log.info("Done. Logs at %s", run_dir)


if __name__ == '__main__':
    _absl_app.run(_main)
