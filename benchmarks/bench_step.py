"""Ablation benchmark: full training-step time for each optimization.

Workloads mirror the real training scripts (same nets, optimizers, loss fns,
batch shapes); eval/logging is excluded.  Each variant is timed for regular
steps and GradNorm-update steps separately, carrying optimizer state forward
like real training (matters for SOAP's one-off init branch).

  2D corrosion (full grid 201x101, Nt=50, FiLM net, Adam):
      sparse vs conv operator
  3D icemelting (128^3 grid, K_core=2000, Nt=64, MLP, SOAP):
      concat vs tensor-split net

Usage:
  python benchmarks/bench_step.py --workload 2d [--tf32]
  python benchmarks/bench_step.py --workload 3d [--precision highest]
"""

import argparse
import importlib.util
import os
import time

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_ROOT, path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_step(loss_fn, optim):
    @eqx.filter_jit
    def step(net, opt_state, args, kwargs):
        loss, aux, grad, new_w = loss_fn(net, *args, **kwargs)
        grad = jax.tree.map(jnp.nan_to_num, grad)
        updates, opt_state = optim.update(grad, opt_state, net)
        net = eqx.apply_updates(net, updates)
        return net, opt_state, loss, new_w
    return step


def _time_steps(step, net, opt_state, make_args, warmup=3, iters=20):
    """Median ms/iter; carries net/opt_state forward like real training."""
    for i in range(warmup):
        net, opt_state, loss, _ = step(net, opt_state, *make_args(i))
        jax.block_until_ready(loss)
    times = []
    for i in range(iters):
        t0 = time.perf_counter()
        net, opt_state, loss, _ = step(net, opt_state, *make_args(100 + i))
        jax.block_until_ready(loss)
        times.append((time.perf_counter() - t0) * 1e3)
    return float(np.median(times))


# ── 2D corrosion workload ───────────────────────────────────────────────────

def bench_2d(tf32=False):
    from pinn.operators import build_D
    from pinn.arch import make_pinn
    from pinn.optimizers import make_optimizer

    losses = _load('examples/corrosion2d/losses.py', 'c2d_losses')
    config = _load('examples/corrosion2d/configs/train_1pit.py',
                   'c2d_config').get_config()

    xs = jnp.linspace(0.0, 1.0, config.sample.Nx)
    ys = jnp.linspace(0.0, 0.5, config.sample.Ny)
    Xv, Yv = jnp.meshgrid(xs, ys, indexing='xy')
    x_base = jnp.stack([Xv.ravel(), Yv.ravel()], axis=-1)
    Nt = config.sample.Nt

    rng = np.random.RandomState(0)
    x_bc_pit = jnp.array(rng.uniform(size=(config.sample.N_bc_dir, 2))
                         * np.array([1.0, 0.0]))
    x_bc_solid = jnp.array(rng.uniform(size=(config.sample.N_bc_solid, 2))
                           * np.array([1.0, 0.5]))
    n_ic = config.sample.N_ic_global + config.sample.N_ic_local
    x_ic = jnp.array(rng.uniform(size=(n_ic, 2)) * np.array([1.0, 0.5]))
    phi0, c0 = losses.ic_values(x_ic, config.physical, config.nondim.L_c,
                                config.pits.entries)

    phys = config.physical
    L_c, T_c = config.nondim.L_c, config.nondim.T_c
    dc0 = phys.c_se - phys.c_le
    nd = (phys.D * T_c / L_c**2, 2.0 * phys.A * phys.L * T_c,
          phys.L * phys.omega_phi * T_c, phys.L * phys.alpha_phi * T_c / L_c**2,
          dc0, phys.c_le)

    c = config.arch
    net = make_pinn(spatial_dim=2, out_dim=2, trunk_type="film",
                     hidden_dim=c.hidden_dim, num_layers=c.num_layers,
                     film_hidden_dim=c.film_hidden_dim,
                     film_num_layers=c.film_num_layers,
                     activation=c.activation,
                     spatial_emb_dim=c.spatial_emb_dim,
                     spatial_scale=c.spatial_scale,
                     temporal_emb_dim=c.temporal_emb_dim,
                     temporal_scale=c.temporal_scale,
                     key=jr.PRNGKey(42))
    optim = make_optimizer(config)

    variants = [
        ("sparse operator", "sparse"),
        ("conv operator",   "conv"),
    ]
    precisions = [("fp32-highest", "highest")]
    if tf32:
        precisions.append(("tensorfloat32", "tensorfloat32"))

    print(f"\n=== 2D corrosion step: {config.sample.Nx}x{config.sample.Ny}"
          f"={len(x_base)} pts, Nt={Nt}, FiLM {c.hidden_dim}x{c.num_layers}, "
          f"Adam ===")
    print(f"{'variant':18s} {'prec':14s} {'regular ms':>11s} {'update ms':>10s}")

    D_cache = {}
    for prec_name, prec in precisions:
        jax.config.update("jax_default_matmul_precision", prec)
        for name, method in variants:
            if method not in D_cache:
                D_cache[method] = build_D(x_base, config.nl.delta,
                                          kernel=config.nl.kernel,
                                          method=method)
            lfn = losses.make_loss_fn(
                x_base, D_cache[method], x_bc_pit, x_bc_solid, nd, x_ic=x_ic,
                alpha=config.train.alpha_gradnorm,
                ac_scale=config.loss.ac_scale, ch_scale=config.loss.ch_scale,
                gradnorm_cap=config.loss.gradnorm_weight_cap, causal=None)
            step = _make_step(lfn, optim)
            opt_state = optim.init(eqx.filter(net, eqx.is_array))
            w_ema = jnp.ones(5)
            bias = jnp.ones(5, dtype=jnp.float32)

            def make_args(i, update_w=False):
                t_batch = jr.uniform(jr.PRNGKey(i), (Nt,),
                                     minval=config.sample.t_min, maxval=1.0)
                args = (t_batch, phi0, c0, jnp.array(1.0), jnp.array(True),
                        jnp.array(True), w_ema, bias, update_w,
                        jnp.array(0.0), None)
                kwargs = dict(x_points=x_base, sub_data=None, core_mask=None)
                return args, kwargs

            ms_reg = _time_steps(step, net, opt_state, lambda i: make_args(i, False))
            ms_upd = _time_steps(step, net, opt_state, lambda i: make_args(i, True))
            print(f"{name:18s} {prec_name:14s} {ms_reg:11.2f} {ms_upd:10.2f}")
    jax.config.update("jax_default_matmul_precision", "highest")


# ── 3D icemelting workload ──────────────────────────────────────────────────

def bench_3d(n_pool=4):
    from pinn.operators import build_D
    from pinn.arch import make_pinn
    from pinn.optimizers import make_optimizer

    losses = _load('examples/icemelting3d/losses.py', 'i3d_losses')
    config = _load('examples/icemelting3d/configs/train_base.py',
                   'i3d_config').get_config()

    Nx = config.sample.Nx
    g = jnp.linspace(0.0, 1.0, Nx)
    Xv, Yv, Zv = jnp.meshgrid(g, g, g, indexing='xy')
    x_base = jnp.stack([Xv.ravel(), Yv.ravel(), Zv.ravel()], axis=-1)
    Nt = config.sample.Nt
    K_core = config.sample.K_core

    phys, ndc = config.physical, config.nondim
    R0_star = phys.R0 / ndc.L_c
    lam_star = phys.lam * ndc.T_c / ndc.L_c
    N1 = phys.M * ndc.T_c / ndc.L_c**2
    N2 = N1 / phys.eps_star**2
    N3 = lam_star / (np.sqrt(2.0) * phys.eps_star)
    denom = np.sqrt(2.0) * phys.eps_star
    xc = np.array(ndc.xc)

    idx_ic = jr.choice(jr.PRNGKey(1), len(x_base), (config.sample.N_ic,),
                       replace=False)
    x_ic = x_base[idx_ic]
    phi0 = losses.ic_values(x_ic, R0_star, xc, denom)

    arch = config.arch
    activation = arch.activation  # "tanh"|"relu"|"gelu" — resolved inside make_pinn
    k_net = jr.PRNGKey(42)
    net = make_pinn(spatial_dim=3, out_dim=1, hidden_dim=arch.hidden_dim,
                    num_layers=arch.num_layers, trunk_type="mlp",
                    activation=activation,
                    spatial_emb_dim=arch.spatial_emb_dim,
                    spatial_scale=arch.spatial_scale,
                    temporal_emb_dim=arch.temporal_emb_dim,
                    temporal_scale=arch.temporal_scale,
                    tensor_split=False, key=k_net)
    net_split = make_pinn(spatial_dim=3, out_dim=1, hidden_dim=arch.hidden_dim,
                           num_layers=arch.num_layers, trunk_type="mlp",
                           activation=activation,
                           spatial_emb_dim=arch.spatial_emb_dim,
                           spatial_scale=arch.spatial_scale,
                           temporal_emb_dim=arch.temporal_emb_dim,
                           temporal_scale=arch.temporal_scale,
                           tensor_split=True, key=k_net)
    optim = make_optimizer(config)

    t0 = time.perf_counter()
    D = build_D(x_base, config.nl.delta, kernel=config.nl.kernel,
                method="sparse")
    print(f"SparseD {Nx}^3 built in {time.perf_counter()-t0:.1f}s "
          f"(k_avg={len(D._vals_np)/D._N:.1f})")

    lfn = losses.make_loss_fn(x_base, D, N1, N2, N3, x_ic=x_ic,
                              alpha=config.train.alpha_gradnorm,
                              gradnorm_cap=config.loss.gradnorm_weight_cap,
                              causal=None)
    pool = D.build_subgraph_pool(K_core, n_pool)
    pool = [(x_base[jnp.array(p[0])], p[1], p[2]) for p in pool]
    N_pad = len(pool[0][0])

    print(f"\n=== 3D icemelting step: {Nx}^3={len(x_base)} pts, "
          f"K_core={K_core} (K_all pad={N_pad}), Nt={Nt}, "
          f"MLP {arch.hidden_dim}x{arch.num_layers}, SOAP ===")
    print(f"{'variant':22s} {'regular ms':>11s} {'update ms':>10s}")

    for name, this_net in [("concat net (baseline)", net),
                           ("tensor-split net", net_split)]:
        step = _make_step(lfn, optim)
        opt_state = optim.init(eqx.filter(this_net, eqx.is_array))
        w_ema = jnp.ones(2)
        bias = jnp.ones(2, dtype=jnp.float32)

        def make_args(i, update_w=False):
            t_batch = jr.uniform(jr.PRNGKey(i), (Nt,),
                                 minval=config.sample.t_min, maxval=1.0)
            x_points, sub_data, core_mask = pool[i % len(pool)]
            args = (t_batch, phi0, w_ema, bias, update_w, jnp.array(0.0),
                    None, x_points, sub_data, core_mask)
            return args, {}

        ms_reg = _time_steps(step, this_net, opt_state,
                             lambda i: make_args(i, False),
                             warmup=3, iters=10)
        ms_upd = _time_steps(step, this_net, opt_state,
                             lambda i: make_args(i, True),
                             warmup=2, iters=5)
        print(f"{name:22s} {ms_reg:11.2f} {ms_upd:10.2f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--workload", choices=["2d", "3d"], default="2d")
    p.add_argument("--tf32", action="store_true",
                   help="add TF32 rows to the 2D benchmark (opt-in)")
    p.add_argument("--precision", default="",
                   help="set jax_default_matmul_precision for the 3D workload "
                        "(default: JAX default, matching the training script)")
    args = p.parse_args()

    print(f"jax {jax.__version__}, device: {jax.devices()[0].device_kind}")
    if args.workload == "2d":
        bench_2d(tf32=args.tf32)
    else:
        if args.precision:
            jax.config.update("jax_default_matmul_precision", args.precision)
            print(f"matmul precision: {args.precision}")
        bench_3d()
