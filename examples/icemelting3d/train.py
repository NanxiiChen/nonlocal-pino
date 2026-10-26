"""3D PINN for ice melting — nonlocal Allen-Cahn with analytical solution.

Usage:
  source .venv/bin/activate
  python examples/icemelting3d/train.py --config=examples/icemelting3d/configs/train_base.py
"""

import json, logging, os, shutil, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from scipy.stats.qmc import LatinHypercube
from tensorboardX import SummaryWriter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from absl import app as _absl_app
from ml_collections import config_flags as _config_flags

from nonlocal_pinn.operators import build_D, SparseD
from nonlocal_pinn.nn.arch import make_pinn
from nonlocal_pinn.io.utils import log_config_text, load_example_losses
from nonlocal_pinn.training.causal import CausalWeightor
from nonlocal_pinn.training.optimizers import make_optimizer
from nonlocal_pinn.training.runtime import with_run_seed
from examples.icemelting3d.phase import FORMULATIONS, wrap_phase
from examples.icemelting3d.evaluation import MeltingEvaluator

log = logging.getLogger(__name__)




# ── Network ────────────────────────────────────────────────────────────────

def _build_network(config, key):
    """Legacy raw builder, also used for historical checkpoint reconstruction."""
    arch = config.arch
    activation = arch.activation   # "tanh"|"relu"|"gelu" — resolved inside make_pinn

    net = make_pinn(spatial_dim=arch.spatial_dim, out_dim=arch.out_dim,
                     trunk_type=arch.type,
                     hidden_dim=arch.hidden_dim, num_layers=arch.num_layers,
                     activation=activation,
                     use_spatial_emb=arch.use_spatial_emb,
                     use_temporal_emb=arch.use_temporal_emb,
                     spatial_emb_dim=arch.spatial_emb_dim,
                     spatial_scale=arch.spatial_scale,
                     temporal_emb_dim=arch.temporal_emb_dim,
                     temporal_scale=arch.temporal_scale,
                     tensor_split=arch.get('tensor_split', False),
                     film_hidden_dim=arch.film_hidden_dim,
                     film_num_layers=arch.film_num_layers,
                     key=key)

    if arch.use_hard_constraint:
        from nonlocal_pinn.nn.constraints import tanh_constraint, ConstrainedForward
        net = ConstrainedForward(net, tanh_constraint)
    return net


def _build_phase_network(config, key):
    return wrap_phase(_build_network(config, key), config)


def _validate_config(config):
    if config.phase.formulation not in FORMULATIONS:
        raise ValueError(f"phase.formulation must be one of {FORMULATIONS}")
    if config.nl.method not in ("sparse", "fd", "local"):
        raise ValueError("Melting training supports nl.method=sparse, fd, or local")
    positive = [config.physical.M, config.physical.eps_star, config.physical.lam,
                config.physical.R0, config.nondim.L_c, config.nondim.T_c, config.nl.delta]
    if not np.all(np.isfinite(positive)) or np.any(np.asarray(positive) <= 0):
        raise ValueError("Physical parameters, scales and nonlocal horizon must be finite and positive")
    centre = np.asarray(config.nondim.xc)
    if centre.shape != (3,) or not np.all(np.isfinite(centre)):
        raise ValueError("nondim.xc must contain three finite coordinates")
    radius = config.physical.R0 / config.nondim.L_c
    if np.any(centre-radius <= 0) or np.any(centre+radius >= 1):
        raise ValueError("Initial sphere must lie strictly inside the unit cube")
    counts = [config.sample.Nx, config.sample.Ny, config.sample.Nz, config.sample.Nt,
              config.sample.K_pool, config.train.iterations, config.log.eval_every,
              config.log.log_every, config.eval.chunk_size, config.eval.full_grid_every]
    if any(not isinstance(v, int) or v <= 0 for v in counts) or min(counts[:3]) < 2:
        raise ValueError("Grid, training, sampling and evaluation counts must be positive integers; grid axes >=2")
    if not 0 <= config.sample.t_min < 1:
        raise ValueError("sample.t_min must be in [0,1)")
    if config.phase.formulation != "raw_phase" and config.arch.use_hard_constraint:
        raise ValueError("IC-only phase modes require arch.use_hard_constraint=False")
    if config.arch.out_dim != 1 or config.arch.spatial_dim != 3:
        raise ValueError("Melting requires a scalar network with three spatial inputs")


def _snapshot_sources(run_dir):
    source = Path(run_dir) / "source"
    shutil.copytree(Path(__file__).parent, source / "icemelting3d",
                    ignore=shutil.ignore_patterns("__pycache__"))
    package = Path(__file__).resolve().parents[2] / "src" / "nonlocal_pinn"
    shutil.copytree(package, source / "nonlocal_pinn", ignore=shutil.ignore_patterns("__pycache__"))
    files = [str(p.relative_to(source)) for p in sorted(source.rglob("*.py"))]
    (source / "files.json").write_text(json.dumps(files, indent=2)+"\n")


# ── Non-dimensional parameters ─────────────────────────────────────────────

def _nondim_params(config):
    phys = config.physical
    nd = config.nondim
    eps_star = phys.eps_star

    R0_star = phys.R0 / nd.L_c
    lam_star = phys.lam * nd.T_c / nd.L_c
    N1 = phys.M * nd.T_c / nd.L_c**2
    N2 = N1 / eps_star**2
    N3 = lam_star / (np.sqrt(2.0) * eps_star)
    denom = np.sqrt(2.0) * eps_star

    return N1, N2, N3, eps_star, denom, lam_star, R0_star


# ── Training step ──────────────────────────────────────────────────────────

def _make_step(loss_fn, optim):
    @eqx.filter_jit
    def step(net, opt_state, t_batch, phi0, w_ema, bias, update_w,
             iface_gamma, eps, x_points, sub_data, core_mask):
        (loss, aux, grad, new_w) = loss_fn(
            net, t_batch, phi0, w_ema, bias, update_w, iface_gamma, eps,
            x_points, sub_data, core_mask)
        # A failed numerical update must remain visible, never silently sanitized.
        aux['finite_update'] = jnp.isfinite(loss) & jnp.all(jnp.stack([
            jnp.all(jnp.isfinite(g)) for g in jax.tree.leaves(eqx.filter(grad, eqx.is_array))]))
        updates, opt_state = optim.update(grad, opt_state, net)
        net = eqx.apply_updates(net, updates)
        return net, opt_state, loss, aux, new_w
    return step


# ── Visualisation ──────────────────────────────────────────────────────────

def _plot_snapshots(net, x_slice, R0_star, lam_star, xc, denom,
                    writer, step, nv=128):
    """Plot φ snapshots at key times on a 2D slice (z=0.5)."""
    losses = load_example_losses(__file__)
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
    fig, axes = plt.subplots(2, n, figsize=(2 * n, 6), squeeze=False)
    for j in range(n):
        phi_p = np.array(phi_pred_all[j]).reshape(nv, nv)
        err = np.array(err_all[j]).reshape(nv, nv)
        im0 = axes[0][j].pcolormesh(phi_p, cmap='coolwarm', vmin=-1, vmax=1,
                                     shading='auto')
        axes[0][j].set_title(rf'$\phi$ PINN  t*={target_times[j]:.2f}')
        axes[0][j].set_aspect('equal')
        axes[0][j].set_axis_off()

        im1 = axes[1][j].pcolormesh(err, cmap='coolwarm', vmin=0, vmax=max_err,
                                     shading='auto')
        axes[1][j].set_title(
            rf'$|\phi - \phi_{{anal}}|$  t*={target_times[j]:.2f}')
        axes[1][j].set_aspect('equal')
        axes[1][j].set_axis_off()

    fig.tight_layout(rect=[0, 0.05, 1, 1])
    cbar_ax = fig.add_axes([0.15, 0.1, 0.7, 0.02])
    fig.colorbar(im1, cax=cbar_ax, orientation='horizontal', label='|error|')
    writer.add_figure('eval/snapshots', fig, step)
    plt.close(fig)


# ── Diagnostic ─────────────────────────────────────────────────────────────

def _diag_residual(net, x_points, sub_data, core_mask, t_eval, N1, N2, N3, writer, step):
    losses = load_example_losses(__file__)
    if sub_data is None:
        phi, dphi_dt, lap = losses._compute_all_local(net, x_points, t_eval)
        diffusion = -lap
    else:
        phi, dphi_dt = losses._compute_all(net, x_points, t_eval)
        row, col, vals, diag = sub_data
        diffusion = diag[:, None] * phi + jax.ops.segment_sum(
            vals[:, None] * phi[col], row, num_segments=len(x_points))
    mask = jnp.ones(len(x_points)) if core_mask is None else core_mask.astype(phi.dtype)

    for ti, t_v in enumerate(t_eval):
        D_phi = diffusion[:, ti]
        t_label = f'{float(t_v):.2f}'

        components = {
            'dphi_dt': dphi_dt[:, ti],
            'N1_Dphi': N1 * D_phi,
            'N2_cubic': N2 * (phi[:, ti]**3 - phi[:, ti]),
            'N3_source': N3 * (1.0 - phi[:, ti]**2),
        }
        total = sum(components.values())
        for name, val in components.items():
            writer.add_scalar(f'diag/{name}/t={t_label}',
                              float(jnp.sqrt(jnp.sum(mask * val**2) / jnp.sum(mask))), step)
        writer.add_scalar(f'diag/total/t={t_label}',
                          float(jnp.sqrt(jnp.sum(mask * total**2) / jnp.sum(mask))), step)



# ── Main ───────────────────────────────────────────────────────────────────

def _main(_):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    config = with_run_seed(FLAGS.config, _SEED.value)
    _validate_config(config)
    _prec = config.train.matmul_precision
    if _prec:
        jax.config.update("jax_default_matmul_precision", _prec)
    run_name = config.log.run_name or datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-%f')
    run_dir = str(Path(config.log.log_dir) / run_name)
    Path(run_dir).mkdir(parents=True, exist_ok=False)
    writer = SummaryWriter(run_dir)
    log.info("Logging to %s", run_dir)
    with open(os.path.join(run_dir, 'config.json'), 'w') as f:
        f.write(json.dumps(json.loads(config.to_json_best_effort()), indent=2))
    log_config_text(writer, config)
    _snapshot_sources(run_dir)

    losses = load_example_losses(__file__)
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
    rng = jr.PRNGKey(config.train.seed)
    k_net, k_ic = jr.split(rng)
    net = _build_phase_network(config, k_net)
    log.info("Network: %s, out_dim=%d, spatial_dim=%d",
             config.arch.type, config.arch.out_dim, config.arch.spatial_dim)
    log.info("Phase formulation: %s (cold start, seed=%d)", config.phase.formulation, config.train.seed)

    # ── Operator ────────────────────────────────────────────────────────
    if config.nl.method == "local":
        D = None  # no pre-built operator; use autograd Laplacian
        log.info("D: local (autograd Laplacian, LHS sampling)")
    elif config.nl.method == "fd":
        from nonlocal_pinn.baselines.fd import FiniteDiff
        D = FiniteDiff(x_base, Nx=Nx, Ny=Ny, Nz=Nz)
        log.info("D: FiniteDiff N=%d k_max=%d", D._N, D._k_max)
    else:
        D = build_D(x_base, config.nl.delta,
                    kernel=config.nl.kernel, method=config.nl.method)
        if isinstance(D, SparseD):
            log.info("D: SparseD N=%d nnz=%d k_avg=%.1f",
                     D._N, len(D._vals), len(D._vals) / D._N)
        else:
            log.info("D: DenseD shape=%s", D.shape)

    # ── IC points ───────────────────────────────────────────────────────
    if config.phase.formulation == "raw_phase":
        if not 0 < config.sample.N_ic <= len(x_base):
            raise ValueError("sample.N_ic must be between 1 and the grid size")
        idx_ic = jr.choice(k_ic, len(x_base), (config.sample.N_ic,), replace=False)
        x_ic = x_base[idx_ic]
        phi0 = ic_values(x_ic, R0_star, xc, denom)
        log.info("Soft IC: N_ic=%d", len(x_ic))
    else:
        x_ic = x_base[:0]
        phi0 = jnp.empty((0,), dtype=x_base.dtype)
        log.info("Hard IC embedded exactly; no IC samples/loss/GradNorm term")

    # ── Sub-graph pool ──────────────────────────────────────────────────
    full_sub_data = None if D is None else (D._row_idx, D._col_idx, D._vals, D._diag)
    K_core = config.sample.K_core
    subgraph_pool = None
    if D is not None and K_core > 0 and K_core < len(x_base) and hasattr(D, 'build_subgraph_pool'):
        K_pool = config.sample.K_pool
        log.info("Pre-building sub-graph pool (%d entries, K_core=%d) ...",
                 K_pool, K_core)
        _t0 = time.perf_counter()
        raw_pool = D.build_subgraph_pool(K_core, K_pool)
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

    # ── LHS sampler (local mode only) ────────────────────────────────────
    lhs_sampler = None
    N_lhs = 0
    if config.nl.method == "local":
        N_lhs = config.sample.N_lhs
        lhs_sampler = LatinHypercube(d=3, seed=config.train.seed)
        log.info("LHS sampler: N_lhs=%d spatial points (resampled each step)", N_lhs)

    # ── Optimizer ───────────────────────────────────────────────────────
    optim = make_optimizer(config)
    opt_state = optim.init(eqx.filter(net, eqx.is_array))

    # ── Causal training ──────────────────────────────────────────────────
    causal = eps_causal = None
    if config.loss.use_causal:
        causal = CausalWeightor(n_chunks=None)  # per-t mode
        eps_causal = jnp.array(config.causal.eps_init)

    # ── Loss function ───────────────────────────────────────────────────
    lfn = make_loss_fn(
        x_base, D, N1, N2, N3, x_ic=x_ic,
        alpha=config.train.alpha_gradnorm,
        gradnorm_cap=config.loss.gradnorm_weight_cap,
        causal=causal,
        formulation=config.phase.formulation,
    )

    # ── Training state ──────────────────────────────────────────────────
    step_fn = _make_step(lfn, optim)
    w_ema = jnp.ones(2)
    bias = jnp.array(config.loss.gradnorm_bias)

    # Eval setup
    N_eval = min(8000, len(x_base))
    idx_eval = jr.choice(jr.PRNGKey(123), len(x_base), (N_eval,), replace=False)
    x_eval = x_base[idx_eval]
    t_eval = jnp.array([0.0, 0.25, 0.5, 0.75, 1.0])
    evaluator = MeltingEvaluator(config, x_base, x_eval, t_eval, run_dir, writer)

    # Slice at z=0.5 for visualisation
    _nv = 128  # viz resolution, independent of training grid
    xs_1d = jnp.linspace(0.0, 1.0, _nv)
    ys_1d = jnp.linspace(0.0, 1.0, _nv)
    Xv_s, Yv_s = jnp.meshgrid(xs_1d, ys_1d, indexing='xy')
    x_slice = jnp.stack([Xv_s.ravel(), Yv_s.ravel(),
                         jnp.full(Xv_s.size, 0.5)], axis=-1)

    # ── Training loop ───────────────────────────────────────────────────
    cfg = config.train
    log_cfg = config.log
    t_min = config.sample.t_min

    initial = evaluator.evaluate(net, 0, full_grid=True)
    log.info("Initial L2: %s", json.dumps(initial))
    eqx.tree_serialise_leaves(os.path.join(run_dir, "model_initial.eqx"), net)
    started = time.perf_counter()
    for it in range(cfg.iterations):
        step = it + 1
        key_t = jr.PRNGKey(it * 3 + 1000)
        t_batch = jr.uniform(key_t, (config.sample.Nt,),
                             minval=t_min, maxval=1.0)
        update_w = (it % cfg.gradnorm_update_every == 0)

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
        elif lhs_sampler is not None:
            x_points = jnp.array(lhs_sampler.random(n=N_lhs), dtype=jnp.float32)
            sub_data = core_mask = None
        else:
            x_points = x_base
            sub_data, core_mask = full_sub_data, None

        net, opt_state, loss_val, aux, w_ema = step_fn(
            net, opt_state, t_batch, phi0, w_ema, bias, update_w,
            iface_gamma, eps_causal, x_points, sub_data, core_mask)
        if not bool(aux['finite_update']):
            raise FloatingPointError(f"Nonfinite loss/gradient at update {step}; run={run_dir}")

        # Causal eps update
        if causal is not None and aux.get('causal/w') is not None:
            eps_causal = causal.update_eps(
                aux['causal/w'], eps_causal,
                config.causal.step_size, config.causal.max_eps,
                config.causal.max_last_weight, config.causal.min_mean_weight)

        # Logging
        if step % log_cfg.log_every == 0:
            writer.add_scalar('loss/total', float(loss_val), step)
            for k, v in aux.items():
                if v is not None and (not isinstance(v, jnp.ndarray) or v.ndim == 0):
                    writer.add_scalar(k, float(v), step)
            if causal is not None and eps_causal is not None:
                writer.add_scalar('causal/eps', float(eps_causal), step)
                if aux.get('causal/w') is not None:
                    w_mean = float(jnp.mean(aux['causal/w']))
                    w_last = float(aux['causal/w'][-1])
                    writer.add_scalar('causal/w_mean', w_mean, step)
                    writer.add_scalar('causal/w_last', w_last, step)
            log.info("%5d: total=%.4e  pde=%.4e  ic=%.4e  w=[%.2f,%.2f]",
                     step, float(loss_val), float(aux['loss/pde']),
                     float(aux['loss/ic']),
                     float(aux['weight/pde']), float(aux['weight/ic']))

        # Evaluation
        full_grid = step % config.eval.full_grid_every == 0 or step == cfg.iterations
        if step % log_cfg.eval_every == 0 or full_grid:
            metrics = evaluator.evaluate(net, step, full_grid=full_grid)
            log.info("L2 update %d: %s", step, json.dumps(metrics))
            _plot_snapshots(net, x_slice, R0_star, lam_star,
                            xc, denom, writer, step)

        # Diagnostic
        if log_cfg.run_diag and step % log_cfg.eval_every == 0:
            _diag_residual(net, x_points, sub_data, core_mask,
                           jnp.array([0.25, 0.5, 0.75, 1.0]),
                           N1, N2, N3, writer, step)

        # Checkpoint
        if log_cfg.save_ckpt and step % log_cfg.eval_every == 0:
            ckpt_path = os.path.join(run_dir, f"ckpt_{step:06d}.eqx")
            eqx.tree_serialise_leaves(ckpt_path, net)
            log.info("  Checkpoint saved: %s", ckpt_path)

    # ── Final save ──────────────────────────────────────────────────────
    final_path = os.path.join(run_dir, "model_final.eqx")
    eqx.tree_serialise_leaves(final_path, net)
    log.info("Final model: %s", final_path)
    summary = {"updates": cfg.iterations, "training_wall_seconds": time.perf_counter()-started,
               "final_metrics": metrics, "cold_start": True,
               "model_path": "model_final.eqx"}
    (Path(run_dir)/"summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    writer.close()


# ── Config ────────────────────────────────────────────────────────────────

_DEFAULT = os.path.join(os.path.dirname(__file__), 'configs', 'train_base.py')
_CONFIG = _config_flags.DEFINE_config_file('config', _DEFAULT, 'Config file')
FLAGS = _absl_app.flags.FLAGS
_SEED = _absl_app.flags.DEFINE_integer('seed', None, 'Run seed; defaults to 42, recorded for reproduction, not strategy tuning')

if __name__ == '__main__':
    _absl_app.run(_main)
