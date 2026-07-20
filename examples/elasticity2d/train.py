"""Training script for 2D elastic plate with elliptical hole — nonlocal PINN.

Left edge clamped (u_x=u_y=0), right edge pulled (u_x=u_load).
Hard constraint on BCs.  D_δ handles free edges.
Only PDE loss — no BC loss terms.

Usage (from project root):
  python examples/elasticity2d/train.py
  python examples/elasticity2d/train.py --config=examples/elasticity2d/configs/train_base.py
  python examples/elasticity2d/train.py --config.nl.method=local --config.log.run_name=local_run
"""

import json, logging, os, time

import jax, jax.numpy as jnp, jax.random as jr
import equinox as eqx
import numpy as np
from scipy.stats.qmc import LatinHypercube
from tensorboardX import SummaryWriter

log = logging.getLogger(__name__)

from pinn.operators import ElasticD
from pinn.arch import make_pinn
from pinn.utils import log_config_text, load_example_losses, setup_run_dir


# ── Config ────────────────────────────────────────────────────────────────────

from absl import app as _absl_app, flags as _absl_flags
from ml_collections import config_flags as _config_flags

_DEFAULT = os.path.join(os.path.dirname(__file__), 'configs', 'train_base.py')
_CONFIG = _config_flags.DEFINE_config_file('config', _DEFAULT, 'Config file')
FLAGS = _absl_flags.FLAGS


# ── Setup helpers ─────────────────────────────────────────────────────────────

def _build_grid(config):
    """Regular grid [0,Lx]×[0,Ly] with elliptical hole excluded."""
    Nx, Ny = config.sample.Nx, config.sample.Ny
    Lx = config.geometry.Lx
    Ly = config.geometry.Ly
    a_x = config.geometry.a_x
    a_y = config.geometry.a_y
    cx, cy = Lx / 2.0, Ly / 2.0

    xs = jnp.linspace(0.0, Lx, Nx)
    ys = jnp.linspace(0.0, Ly, Ny)
    Xv, Yv = jnp.meshgrid(xs, ys, indexing='xy')
    x_full = jnp.stack([Xv.ravel(), Yv.ravel()], axis=-1)

    # Exclude points inside the elliptical hole: ((x-cx)/a_x)²+((y-cy)/a_y)² >= 1
    dx = (x_full[:, 0] - cx) / a_x
    dy = (x_full[:, 1] - cy) / a_y
    mask = dx**2 + dy**2 >= 1.0
    x_points = x_full[mask]
    return jnp.array(x_points)


def _build_network(config, key):
    """Build PINN with hard constraint: u_x=(x/L)*u_load + x(L-x)*N_x,
    u_y = x*N_y."""
    c = config.arch
    Lx = config.geometry.Lx
    Ly = config.geometry.Ly
    u_load = config.geometry.u_load

    net = make_pinn(spatial_dim=2, out_dim=c.out_dim, trunk_type=c.type,
                     hidden_dim=c.hidden_dim, num_layers=c.num_layers,
                     activation=c.activation,
                     use_spatial_emb=c.use_spatial_emb,
                     use_temporal_emb=False, static=True,
                     multi_head=c.multi_head,
                     spatial_emb_dim=c.spatial_emb_dim,
                     spatial_scale=c.spatial_scale,
                     tensor_split=False,
                     film_hidden_dim=c.film_hidden_dim,
                     film_num_layers=c.film_num_layers,
                     key=key)

    if c.use_hard_constraint:
        from pinn.constraints import ConstrainedForward

        def constraint(raw, x, _t):
            """Hard constraint: u_x = (x/Lx)*u_load + x(Lx-x)*N_x,
            u_y = x * u_load * N_y  (scale so N_y ~ O(1))."""
            x_coord = x[..., 0]
            u_x_ref = (x_coord / Lx) * u_load
            u_y_ref = jnp.zeros_like(x_coord)
            d_x = x_coord * (Lx - x_coord) * u_load
            d_y = x_coord * (Lx - x_coord) * u_load
            out_x = u_x_ref + d_x * raw[..., 0]
            out_y = u_y_ref + d_y * raw[..., 1]
            return jnp.stack([out_x, out_y], axis=-1)

        net = ConstrainedForward(net, constraint, pass_input=True)
    return net


def _build_D(config, x_all):
    """Build ElasticD operator (fully nonlocal Navier-Cauchy)."""
    if config.nl.method == "local":
        return None
    return ElasticD(x_all, config.nl.delta, kernel=config.nl.kernel)


def _build_optimizer(config):
    """Create optimizer from config."""
    from pinn.optimizers import make_optimizer
    return make_optimizer(config)


def _sample_bc_points(config):
    """Sample Neumann BC points for local mode."""
    rng = np.random.RandomState(42)
    Lx = config.geometry.Lx
    Ly = config.geometry.Ly
    a_x = config.geometry.a_x
    a_y = config.geometry.a_y
    cx, cy = Lx / 2, Ly / 2
    n_bc = 200

    xs_t = rng.uniform(0, Lx, n_bc)
    xs_b = rng.uniform(0, Lx, n_bc)
    thetas = rng.uniform(0, 2 * np.pi, n_bc)

    top = jnp.column_stack([xs_t, np.full(n_bc, Ly)])
    bottom = jnp.column_stack([xs_b, np.zeros(n_bc)])
    hole = jnp.column_stack([cx + a_x * np.cos(thetas),
                              cy + a_y * np.sin(thetas)])
    return top, bottom, hole, (cx, cy, a_x, a_y)


# ── Training step ─────────────────────────────────────────────────────────────

def _make_step(loss_fn, optim):
    @eqx.filter_jit
    def step(net, opt_state, w_ema, bias, update_w,
             x_points=None):
        (loss, aux, grad, new_w) = loss_fn(
            net, w_ema, bias, update_w, x_points=x_points)
        grad = jax.tree.map(jnp.nan_to_num, grad)
        updates, opt_state = optim.update(grad, opt_state, net)
        net = eqx.apply_updates(net, updates)
        return net, opt_state, loss, aux, new_w
    return step


# ── Evaluation ────────────────────────────────────────────────────────────────

def _eval_fem(net, fem_data, writer, step, config):
    """Compare PINN vs FEM reference.

    Evaluates on a regular grid via interpolation — avoids triangulation
    artefacts near the hole.  Also checks the hard constraint is satisfied
    at the boundaries.
    """
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as onp

    coords_fem = jnp.array(fem_data['coords'])
    u_x_fem = jnp.array(fem_data['u_x'])
    u_y_fem = jnp.array(fem_data['u_y'])
    u_pred_fem = jax.vmap(net, in_axes=(0, None))(coords_fem, jnp.array(0.0))
    rel_ux = float(jnp.sqrt(jnp.mean((u_pred_fem[:, 0] - u_x_fem)**2))
                    / jnp.sqrt(jnp.mean(u_x_fem**2) + 1e-10))
    rel_uy = float(jnp.sqrt(jnp.mean((u_pred_fem[:, 1] - u_y_fem)**2))
                    / jnp.sqrt(jnp.mean(u_y_fem**2) + 1e-10))

    writer.add_scalar('eval/rel_L2_u_x', rel_ux, step)
    writer.add_scalar('eval/rel_L2_u_y', rel_uy, step)

    # Evaluate on a regular grid for clean plots (no triangulation artefacts)
    Lx = config.geometry.Lx
    Ly = config.geometry.Ly
    a_x = config.geometry.a_x
    a_y = config.geometry.a_y
    n_grid = 100
    xs = jnp.linspace(0, Lx, n_grid)
    ys = jnp.linspace(0, Ly, n_grid)
    Xv, Yv = jnp.meshgrid(xs, ys, indexing='xy')
    x_grid_flat = jnp.stack([Xv.ravel(), Yv.ravel()], axis=-1)

    # PINN prediction on regular grid
    u_grid = jax.vmap(net, in_axes=(0, None))(x_grid_flat, jnp.array(0.0))
    ux_pinn = onp.array(u_grid[:, 0].reshape(n_grid, n_grid))
    uy_pinn = onp.array(u_grid[:, 1].reshape(n_grid, n_grid))

    # Interpolate FEM → regular grid
    from scipy.interpolate import griddata
    fem_pts = onp.array(coords_fem)
    ux_fem_grid = griddata(fem_pts, onp.array(u_x_fem),
                           (onp.array(Xv), onp.array(Yv)), method='linear')
    uy_fem_grid = griddata(fem_pts, onp.array(u_y_fem),
                           (onp.array(Xv), onp.array(Yv)), method='linear')

    err_ux = ux_pinn - ux_fem_grid
    err_uy = uy_pinn - uy_fem_grid

    # Mask elliptical hole interior for display
    dx = (onp.array(Xv) - Lx/2) / a_x
    dy = (onp.array(Yv) - Ly/2) / a_y
    hole = dx**2 + dy**2 < 1.0
    for arr in [ux_pinn, uy_pinn, ux_fem_grid, uy_fem_grid, err_ux, err_uy]:
        arr[hole] = onp.nan

    # 2×3 plot
    fig, axes = plt.subplots(2, 3, figsize=(16, 6))
    extent = [0, Lx, 0, Ly]
    panels = [
        (ux_fem_grid, 'u_x FEM'), (ux_pinn, 'u_x PINN'), (err_ux, 'u_x error'),
        (uy_fem_grid, 'u_y FEM'), (uy_pinn, 'u_y PINN'), (err_uy, 'u_y error'),
    ]
    for ax, (f, title) in zip(axes.flat, panels):
        im = ax.imshow(f, extent=extent, origin='lower',
                       cmap='coolwarm', aspect='equal',
                       interpolation='bilinear', rasterized=True)
        ax.set_title(title); ax.set_xlabel('x'); ax.set_ylabel('y')
        plt.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    writer.add_figure('eval/snapshots', fig, step)
    plt.close(fig)

    # Check hard constraint at edges
    left_mask = coords_fem[:, 0] < 0.001
    right_mask = coords_fem[:, 0] > Lx - 0.001
    u_left = u_pred_fem[left_mask]
    u_right = u_pred_fem[right_mask]
    bc_err_ux = float(jnp.sqrt(jnp.mean(u_left[:, 0]**2) +
                                jnp.mean(u_left[:, 1]**2)))
    bc_err_r = float(jnp.sqrt(jnp.mean((u_right[:, 0] -
                                        jnp.array(config.geometry.u_load))**2)))
    writer.add_scalar('eval/bc_left_err', bc_err_ux, step)
    writer.add_scalar('eval/bc_right_err', bc_err_r, step)

    return rel_ux, rel_uy


# ── Main ──────────────────────────────────────────────────────────────────────

def _main(_):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    config = FLAGS.config
    _prec = config.train.get('matmul_precision', '')
    if _prec:
        jax.config.update("jax_default_matmul_precision", _prec)
    rng = jr.PRNGKey(42)
    k_net, _ = jr.split(rng)

    loss_mod = load_example_losses(__file__)
    make_loss_fn = loss_mod.make_loss_fn

    # ── dirs & logging ────────────────────────────────────────────────────
    run_dir = setup_run_dir(config)
    writer = SummaryWriter(run_dir)
    log.info("Logging to %s", run_dir)
    with open(os.path.join(run_dir, 'config.json'), 'w') as f:
        f.write(json.dumps(json.loads(config.to_json_best_effort()), indent=2))
    log_config_text(writer, config)

    # ── spatial grid (hole excluded) ──────────────────────────────────────
    x_base = _build_grid(config)
    log.info("Grid: %d points (ellipse a_x=%.2f, a_y=%.2f excluded)", len(x_base),
             config.geometry.a_x, config.geometry.a_y)

    # ── network + D + optimizer ───────────────────────────────────────────
    net = _build_network(config, k_net)
    D = _build_D(config, x_base)
    lam, mu = config.physical.lam, config.physical.mu

    log.info("lam*=%.4f, mu*=%.4f, delta=%.3f, D method=%s",
             lam, mu, config.nl.delta, config.nl.method)
    if D is not None:
        log.info("ElasticD: N=%d, k_avg=%.1f", D._N, D._k_avg)
    else:
        log.info("Local (all-AD) mode — Neumann BCs active")

    # ── Neumann BC points (local mode only) ───────────────────────────────
    local_mode = (D is None)
    if local_mode:
        x_bc_top, x_bc_bottom, x_bc_hole, (cx, cy, a_x, a_y) = _sample_bc_points(config)
        log.info("BC points: top/bottom/hole = %d/%d/%d",
                 len(x_bc_top), len(x_bc_bottom), len(x_bc_hole))
        bc_kw = dict(x_bc_top=x_bc_top, x_bc_bottom=x_bc_bottom,
                     x_bc_hole=x_bc_hole, hole_center=(cx, cy, a_x, a_y),
                     alpha=config.train.alpha_gradnorm)
    else:
        bc_kw = {}

    lfn = make_loss_fn(x_base if not local_mode else None, D, lam, mu,
                       gradnorm_cap=config.loss.gradnorm_weight_cap,
                       **bc_kw)
    optim = _build_optimizer(config)
    opt_state = optim.init(eqx.filter(net, eqx.is_array))
    step_fn = _make_step(lfn, optim)

    # ── FEM data for eval ─────────────────────────────────────────────────
    fem_data = np.load(config.fem.data_path)

    # ── training loop ─────────────────────────────────────────────────────
    N_LOSS = 4 if local_mode else 1
    w_ema = jnp.ones(N_LOSS)
    default_bias = jnp.array(config.loss.gradnorm_bias
                             if len(config.loss.gradnorm_bias) == N_LOSS
                             else [1.0] * N_LOSS, dtype=jnp.float32)
    t_start = time.perf_counter()

    # LHS collocation sampler (local mode only)
    Lx = config.geometry.Lx
    Ly = config.geometry.Ly
    a_x = config.geometry.a_x
    a_y = config.geometry.a_y
    lhs_sampler = None
    if local_mode:
        N_lhs = config.sample.get('N_lhs', 2000)
        lhs_sampler = LatinHypercube(d=2, seed=42)
        log.info("LHS sampler: N_lhs=%d (resampled each step)", N_lhs)

    gradnorm_every = config.train.get('gradnorm_update_every', 1)
    for it in range(config.train.iterations):
        update_w = (it % gradnorm_every == 0)

        if lhs_sampler is not None:
            # Oversample + truncate → fixed size (avoid JIT retrace)
            raw = lhs_sampler.random(n=int(N_lhs * 1.3))
            pts = np.column_stack([raw[:, 0] * Lx, raw[:, 1] * Ly])
            dx = (pts[:, 0] - Lx/2) / a_x
            dy = (pts[:, 1] - Ly/2) / a_y
            pts = pts[dx**2 + dy**2 >= 1.0][:N_lhs]
            x_points = jnp.array(pts)
        else:
            x_points = x_base

        net, opt_state, loss_val, aux, w_ema = step_fn(
            net, opt_state, w_ema, default_bias, update_w,
            x_points=x_points)

        if it % config.log.log_every == 0:
            elapsed = time.perf_counter() - t_start
            writer.add_scalar('loss/total', float(loss_val), it)
            for k, v in aux.items():
                if v is not None and (not isinstance(v, jnp.ndarray) or v.ndim == 0):
                    writer.add_scalar(k, float(v), it)
            lr = config.train.lr_init * config.train.lr_decay_rate**(it // config.train.lr_decay_every)
            writer.add_scalar('train/lr', float(max(lr, config.train.lr_min)), it)
            if local_mode:
                log.info("[%5d] loss=%.3e  pde=%.3e  top=%.3e  bot=%.3e  hole=%.3e  (%.1fs)",
                         it, float(loss_val), float(aux['loss/pde']),
                         float(aux['loss/bc_top']), float(aux['loss/bc_bottom']),
                         float(aux['loss/bc_hole']), elapsed)
            else:
                log.info("[%5d] loss=%.3e  pde=%.3e  (%.1fs)",
                         it, float(loss_val), float(aux['loss/pde']), elapsed)

        if it % config.log.eval_every == 0:
            rel_ux, rel_uy = _eval_fem(net, fem_data, writer, it, config)
            log.info("--- eval @ %d: rel-L2 u_x=%.4f  u_y=%.4f ---",
                     it, rel_ux, rel_uy)

            if config.log.save_ckpt:
                ckpt_path = os.path.join(run_dir, f'ckpt_{it:06d}.eqx')
                eqx.tree_serialise_leaves(ckpt_path, net)
                log.info("  Saved checkpoint: %s", ckpt_path)

    writer.close()
    log.info("Done. Logs at %s", run_dir)


if __name__ == '__main__':
    _absl_app.run(_main)
