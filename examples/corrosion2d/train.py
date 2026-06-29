"""Training script for full-MLP PINN on 2D KKS phase-field corrosion.

Usage (from project root):
  python examples/corrosion2d/train.py
  python examples/corrosion2d/train.py --config=examples/corrosion2d/configs/train_diffusion.py
  python examples/corrosion2d/train.py --config.train.iterations=1000 --config.nl.delta=0.05
"""

import importlib.util as _iu
import logging, os, time

import jax, jax.numpy as jnp, jax.random as jr

jax.config.update("jax_default_matmul_precision", "highest")
import equinox as eqx
import numpy as np
from tensorboardX import SummaryWriter

log = logging.getLogger(__name__)

from pinn.operators import build_D
from pinn.arch import make_pinn, make_film_pinn


def _load_losses():
    """Load losses_full.py from the same directory as this script."""
    path = os.path.join(os.path.dirname(__file__), 'losses.py')
    spec = _iu.spec_from_file_location('corrosion2d_losses', path)
    mod = _iu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── Config ──────────────────────────────────────────────────────────────────

from absl import app as _absl_app, flags as _absl_flags
from ml_collections import config_flags as _config_flags

_DEFAULT = os.path.join(os.path.dirname(__file__), 'configs', 'train_1pit.py')
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
    hc = None
    if c.use_hard_constraint:
        from pinn.arch.hard_constraint import HardConstraint
        phys = config.physical
        hc = HardConstraint(phys.c_se, phys.c_le)
    if c.type == "film":
        return make_film_pinn(spatial_dim=2, out_dim=2, hidden_dim=c.hidden_dim,
                              num_layers=c.num_layers,
                              film_hidden_dim=c.film_hidden_dim,
                              activation=c.activation,
                              spatial_emb_dim=c.spatial_emb_dim,
                              spatial_scale=c.spatial_scale,
                              temporal_emb_dim=c.temporal_emb_dim,
                              temporal_scale=c.temporal_scale,
                              use_fourier=c.use_spatial_emb or c.use_temporal_emb,
                              hard_constraint=hc, key=key)
    return make_pinn(spatial_dim=2, out_dim=2, hidden_dim=c.hidden_dim,
                     num_layers=c.num_layers, trunk_type=c.type,
                     activation=c.activation,
                     spatial_emb_dim=c.spatial_emb_dim,
                     spatial_scale=c.spatial_scale,
                     temporal_emb_dim=c.temporal_emb_dim,
                     temporal_scale=c.temporal_scale,
                     use_fourier=c.use_spatial_emb or c.use_temporal_emb,
                     hard_constraint=hc, dirichlet_bc=None, key=key)


def _build_D(config, x_all):
    return build_D(x_all, config.nl.delta, kernel=config.nl.kernel, method=config.nl.method)


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


def _bc_info(pits, N_dir=1000, N_solid=500):
    """Sample BC points for 2D corrosion geometry.

    Dirichlet (pit mouth): y=0, |x-cx| < R*0.5 → φ=0, c=0.
    Solid boundaries (left/right/top): x=0, x=1, y=0.5 → φ=1, c=1.
    Pure random sampling — no grid points needed (boundaries are now separate).
    """
    rng = np.random.RandomState(42)
    pit_mouth_x = [(float(cx) - float(R)*0.5, float(cx) + float(R)*0.5) for cx, cy, R in pits]

    # Pit mouth: sample uniformly from each pit's mouth range on y=0
    pit_points = []
    n_each = N_dir // len(pits) + 1
    for lo, hi in pit_mouth_x:
        xs = rng.uniform(lo, hi, n_each)
        pit_points.append(np.column_stack([xs, np.zeros_like(xs)]))
    x_bc_pit = jnp.array(np.vstack(pit_points)[:N_dir])

    # Solid edges: sample uniformly from left (x=0), right (x=1), top (y=0.5)
    solid_points = []
    n_each_edge = N_solid // 3 + 1
    for edge in range(3):
        if edge == 0:   # left: x=0, y in [0, 0.5]
            xs = np.zeros(n_each_edge)
            ys = rng.uniform(0.0, 0.5, n_each_edge)
        elif edge == 1: # right: x=1, y in [0, 0.5]
            xs = np.ones(n_each_edge)
            ys = rng.uniform(0.0, 0.5, n_each_edge)
        else:           # top: y=0.5, x in [0, 1]
            xs = rng.uniform(0.0, 1.0, n_each_edge)
            ys = np.full(n_each_edge, 0.5)
        solid_points.append(np.column_stack([xs, ys]))
    x_bc_solid = jnp.array(np.vstack(solid_points)[:N_solid])

    return x_bc_pit, x_bc_solid


def _sample_ic_points(pits, N_global=2000, N_local=2000):
    """Sample IC points: global uniform + local oversampling near pit interface.

    Global: N_global uniform random points in [0,1]×[0,0.5].
    Local: N_local uniform random points in 2R×2R rectangle around each pit.
    """
    rng = np.random.RandomState(42)

    # Global uniform samples
    xs_g = rng.uniform(0.0, 1.0, N_global)
    ys_g = rng.uniform(0.0, 0.5, N_global)
    points = [np.column_stack([xs_g, ys_g])]

    # Local oversampling near each pit
    n_each = N_local // len(pits) + 1
    for cx, cy, R in pits:
        x_lo = max(float(cx) - 2.0 * float(R), 0.0)
        x_hi = min(float(cx) + 2.0 * float(R), 1.0)
        y_lo = max(float(cy) - 2.0 * float(R), 0.0)
        y_hi = min(float(cy) + 2.0 * float(R), 0.5)
        xs = rng.uniform(x_lo, x_hi, n_each)
        ys = rng.uniform(y_lo, y_hi, n_each)
        points.append(np.column_stack([xs, ys]))

    return jnp.array(np.vstack(points)[:N_global + N_local])


# ── Training step ───────────────────────────────────────────────────────────

def _make_step(loss_fn, optim):
    @eqx.filter_jit
    def step(net, opt_state, t_batch, phi0, c0, eps, train_ac, train_ch, w_ema, bias, update_w, iface_gamma):
        (loss, aux, grad, new_w) = loss_fn(net, t_batch, phi0, c0, eps, train_ac, train_ch, w_ema, bias, update_w, iface_gamma)
        grad = jax.tree.map(jnp.nan_to_num, grad)
        updates, opt_state = optim.update(grad, opt_state, net)
        net = eqx.apply_updates(net, updates)
        return net, opt_state, loss, aux, new_w
    return step


# ── Evaluation ──────────────────────────────────────────────────────────────

def _eval_fem(net, fem_data, writer, step):
    X = jnp.array(fem_data['X'])           # (ny+1, nx+1)
    Y = jnp.array(fem_data['Y'])
    t_fem = jnp.array(fem_data['time'])    # (n_snaps,)
    phi_fem = jnp.array(fem_data['phi'])   # (n_snaps, ny+1, nx+1)
    c_fem = jnp.array(fem_data['c'])

    x_fem = jnp.stack([X.ravel(), Y.ravel()], axis=-1)  # (N_fem, 2)
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

    # Snapshots at key times: φ prediction + absolute error
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
        axes[0][j].pcolormesh(X, Y, phi_pred[i], cmap='coolwarm',
                              vmin=0, vmax=1, shading='auto')
        axes[0][j].set_title(rf'$\phi$ PINN  t*={t_fem[i]:.2f}')
        axes[0][j].set_aspect('equal')

        im1 = axes[1][j].pcolormesh(X, Y, err_phi[i], cmap='coolwarm',
                                     vmin=0, vmax=max_err, shading='auto')
        axes[1][j].set_title(rf'$|\phi - \phi_{{FEM}}|$  t*={t_fem[i]:.2f}')
        axes[1][j].set_aspect('equal')

    fig.tight_layout(rect=[0, 0.1, 1, 1])
    cbar_ax = fig.add_axes([0.15, 0.1, 0.7, 0.025])
    fig.colorbar(im1, cax=cbar_ax, orientation='horizontal', label='|error|')
    writer.add_figure('eval/snapshots', fig, step)
    plt.close(fig)

    return rel_phi, rel_c


def _diag_residual(net, x_all, D, t_eval, nd_params, writer, step):
    """Break down AC residual components on training grid."""
    N_CH, N_AC1, N_AC2, N_AC3, dc0, c_le = nd_params
    _h = _load_losses()._h
    _hp = _load_losses()._hp
    _gp = _load_losses()._gp
    _compute_all = _load_losses()._compute_all

    phi, c, dphi_dt, dc_dt = _compute_all(net, x_all, t_eval)  # (N_x, N_t)
    nl_phi = D @ phi    # (N_x, N_t)
    mu = c - _h(phi) * dc0 - c_le       # (N_x, N_t)
    nl_mu = D @ mu                       # (N_x, N_t)

    # Transpose to (N_t, N_x) for per-component analysis
    phi_t, c_t, dphi_dt_t = phi.T, c.T, dphi_dt.T
    nl_phi_t = nl_phi.T

    r_t = dphi_dt_t
    r_D = N_AC3 * nl_phi_t
    r_g = N_AC2 * _gp(phi_t)
    r_drv = -N_AC1 * (c_t - _h(phi_t) * dc0 - c_le) * dc0 * _hp(phi_t)

    for name, val in [('ac/dphi_dt', r_t), ('ac/N_AC3_Dphi', r_D),
                       ('ac/N_AC2_gp', r_g), ('ac/driving', r_drv),
                       ('ac/total', r_t + r_D + r_g + r_drv)]:
        rms = float(jnp.sqrt(jnp.mean(val**2)))
        writer.add_scalar(f'diag/{name}_rms', rms, step)

    r_ch = dc_dt.T + N_CH * nl_mu.T
    writer.add_scalar('diag/ch_total_rms', float(jnp.sqrt(jnp.mean(r_ch**2))), step)

    log.info("  AC diag: dt=%.2e  Dphi=%.2e  gp=%.2e  drv=%.2e",
             float(jnp.sqrt(jnp.mean(r_t**2))),
             float(jnp.sqrt(jnp.mean(r_D**2))),
             float(jnp.sqrt(jnp.mean(r_g**2))),
             float(jnp.sqrt(jnp.mean(r_drv**2))))


# ── Main ────────────────────────────────────────────────────────────────────

def _main(_):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    config = FLAGS.config
    rng = jr.PRNGKey(42)
    k_net, k_samp = jr.split(rng)

    # Load loss module
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
    x_base = jnp.stack([Xv.ravel(), Yv.ravel()], axis=-1)  # (Nx*Ny, 2)

    delta = config.nl.delta
    x_bc_pit, x_bc_solid = _bc_info(config.pits.entries,
                                     config.sample.N_bc_dir,
                                     config.sample.N_bc_solid)
    log.info("PDE: %d×%d=%d points, delta=%.2f, bc_pit=%d, bc_solid=%d",
             config.sample.Nx, config.sample.Ny, len(x_base), delta,
             len(x_bc_pit), len(x_bc_solid))

    net = _build_network(config, k_net)
    D = _build_D(config, x_base)
    nd_params = _nondim_params(config)
    causal = None
    causal_eps_ac = jnp.array(1.0)
    causal_eps_ch = jnp.array(1.0)
    if config.loss.use_causal:
        from pinn.causal import CausalWeightor
        causal = CausalWeightor(n_chunks=config.causal.n_chunks)
        causal_eps_ac = jnp.array(config.causal.eps_init)
        causal_eps_ch = jnp.array(config.causal.eps_init)
    # --- IC point sampling ---
    x_ic = _sample_ic_points(config.pits.entries,
                             config.sample.N_ic_global,
                             config.sample.N_ic_local)
    log.info("IC points: %d (global=%d + local=%d)",
             len(x_ic), config.sample.N_ic_global, config.sample.N_ic_local)

    lfn = make_loss_fn(x_base, D, x_bc_pit, x_bc_solid, nd_params,
                       x_ic=x_ic,
                       alpha=config.train.alpha_gradnorm,
                       ac_scale=config.loss.ac_scale,
                       ch_scale=config.loss.ch_scale,
                       gradnorm_cap=config.loss.gradnorm_weight_cap,
                       causal=causal)
    optim = _build_optimizer(config)
    opt_state = optim.init(eqx.filter(net, eqx.is_array))
    step_fn = _make_step(lfn, optim)

    # --- initial condition ---
    phi0, c0 = ic_values(x_ic, config.physical, config.nondim.L_c,
                         config.pits.entries)
    log.info("IC: phi min=%.3f max=%.3f  c min=%.3f max=%.3f",
             float(jnp.min(phi0)), float(jnp.max(phi0)),
             float(jnp.min(c0)), float(jnp.max(c0)))

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
    w_ema = jnp.ones(5)  # ac, ch, bc_pit, bc_solid, ic
    t_start = time.perf_counter()

    for it in range(1, config.train.iterations + 1):
        key_t = jr.fold_in(k_samp, it)
        t_batch = jr.uniform(key_t, (config.sample.Nt,),
                             minval=config.sample.t_min, maxval=1.0)

        eps = jnp.minimum(causal_eps_ac, causal_eps_ch)
        if config.loss.use_stagger:
            phase = (it // config.loss.stagger_switch_every) % 2
            train_ac = jnp.array(phase == 0)
            train_ch = jnp.array(phase == 1)
        else:
            train_ac = jnp.array(True)
            train_ch = jnp.array(True)
        update_w = (it % config.train.gradnorm_update_every == 0)
        # Interface weight decay
        gamma_decay = config.loss.iface_gamma_decay_steps
        if gamma_decay > 0 and it <= gamma_decay:
            iface_gamma = jnp.array(config.loss.iface_gamma_init
                                    * (1.0 - it / gamma_decay))
        else:
            iface_gamma = jnp.array(config.loss.iface_gamma_decay_to)
        net, opt_state, loss_val, aux, w_ema = step_fn(
            net, opt_state, t_batch, phi0, c0, eps, train_ac, train_ch, w_ema,
            get_bias(it), update_w, iface_gamma)

        if it % config.log.log_every == 0:
            elapsed = time.perf_counter() - t_start
            writer.add_scalar('loss/total', float(loss_val), it)
            for k, v in aux.items():
                if v is not None and (not isinstance(v, jnp.ndarray) or v.ndim == 0):
                    writer.add_scalar(f'loss/{k}', float(v), it)
            lr = config.train.lr_init * config.train.lr_decay_rate**(it // config.train.lr_decay_every)
            writer.add_scalar('train/lr', float(max(lr, config.train.lr_min)), it)
            # Interface indicator mean (phase-field: 4φ(1-φ), peaks at interface)
            if config.log.run_diag and config.loss.iface_gamma_init > 0:
                phi_t0 = jax.vmap(net, in_axes=(0, None))(x_base, jnp.array(0.5))[:, 0]
                iface_mean = float(jnp.mean(4.0 * phi_t0 * (1.0 - phi_t0)))
                writer.add_scalar('diag/iface_indicator_mean', iface_mean, it)
            log.info("[%5d] loss=%.3e  ac=%.3e ch=%.3e bc_p=%.3e bc_s=%.3e ic=%.3e  "
                     "w=[%.1f,%.1f,%.1f,%.1f,%.0f]  (%.1fs)",
                     it, float(loss_val), float(aux['loss_ac']),
                     float(aux['loss_ch']),
                     float(aux['loss_bc_pit']), float(aux['loss_bc_solid']),
                     float(aux['loss_ic']),
                     float(aux['weight_ac']), float(aux['weight_ch']),
                     float(aux['weight_bc_pit']), float(aux['weight_bc_solid']),
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

            if config.log.run_diag:
                t_eval = jnp.linspace(config.sample.t_min, 1.0, 50)
                _diag_residual(net, x_base, D, t_eval, nd_params, writer, it)

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
