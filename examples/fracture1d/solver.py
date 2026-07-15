"""1D phase-field fracture FEniCS solver — non-dimensional, staggered.

AT2 model (Miehe et al. 2010) on a 1D bar x ∈ [-0.5, 0.5].
Displacement-controlled tension with a small central imperfection to
trigger localisation.  Staggered scheme: elasticity → history update →
phase-field, iterated to convergence per load step.

Outputs: .npz results + analytical comparison plots.

Environment:
  conda activate fenicsenv

Usage (from project root):
  python examples/fracture1d/solver.py
  python examples/fracture1d/solver.py --config=examples/fracture1d/configs/solver_base.py
  python examples/fracture1d/solver.py --config.loading.du=2e-5 --config.loading.u_final=0.06
"""

import gc, os, sys

# Ensure DOLFIN pkg-config is discoverable (see memory pitfalls-fenics-mpi.md).
_pkgconfig_dir = os.path.join(sys.prefix, 'lib', 'pkgconfig')
if os.path.isdir(_pkgconfig_dir):
    os.environ.setdefault('PKG_CONFIG_PATH', '')
    if _pkgconfig_dir not in os.environ['PKG_CONFIG_PATH']:
        os.environ['PKG_CONFIG_PATH'] = (
            _pkgconfig_dir + ':' + os.environ['PKG_CONFIG_PATH']
        )

import numpy as np
from absl import app as _absl_app
from ml_collections import config_flags as _config_flags

from fenics import (
    Constant,
    DirichletBC,
    Function,
    FunctionSpace,
    IntervalMesh,
    TestFunction,
    TrialFunction,
    assemble,
    dx,
    errornorm,
    grad,
    near,
    project,
    solve,
)


# ── Non-dimensional analytical solutions ─────────────────────────────────────

def analytical_homogeneous_damage(eps, N_l, N_G):
    """Homogeneous damage: d = ε² / (ε² + N_G/N_l)."""
    alpha = N_G / N_l
    return eps**2 / (eps**2 + alpha)


def analytical_homogeneous_stress(eps, N_l, N_G):
    """Homogeneous stress: σ = (1-d)² ε with d from above."""
    d = analytical_homogeneous_damage(eps, N_l, N_G)
    return (1.0 - d)**2 * eps


def analytical_crack_profile(x, xc, N_l):
    """Fully-developed crack profile: d(x) = exp(-|x-xc|/N_l)."""
    return np.exp(-np.abs(x - xc) / N_l)


# ── Solver ───────────────────────────────────────────────────────────────────

def solve_fracture(config):
    """Run the 1D staggered phase-field fracture solver."""
    phys = config.physical
    L0 = config.nondim.L0

    # Non-dimensional parameters
    N_l = phys.l0 / L0
    N_G = phys.Gc / (phys.E * L0)
    alpha_nd = N_G / N_l          # = G_c / (l_0 E), controls damage onset

    eps_c = np.sqrt(alpha_nd)     # critical strain (analytical)
    sigma_c = np.sqrt(alpha_nd) / 4.0  # peak homogeneous stress

    print(f"N_l = {N_l:.6f}")
    print(f"N_G = {N_G:.6e}")
    print(f"alpha = N_G/N_l = {alpha_nd:.6e}")
    print(f"eps_c (analytical) = {eps_c:.6f}")
    print(f"sigma_c (analytical) = {sigma_c:.6f}")

    # ── Mesh & function spaces ─────────────────────────────────────────────
    nx = config.mesh.nx
    mesh = IntervalMesh(nx, -0.5, 0.5)

    Vu = FunctionSpace(mesh, 'CG', 1)   # displacement u
    Vd = FunctionSpace(mesh, 'CG', 1)   # damage d
    V0 = FunctionSpace(mesh, 'CG', 1)   # auxiliary scalar fields

    # ── Functions ──────────────────────────────────────────────────────────
    u_new = Function(Vu)
    u_old = Function(Vu)
    d_new = Function(Vd)
    d_old = Function(Vd)
    H_field = Function(V0)      # history field H(x) = max ε²/2
    H_field.assign(Constant(0.0))

    # Spatial imperfection: reduced G_c near x=0
    x_coords = V0.tabulate_dof_coordinates()[:, 0]
    imp_amp = config.imperfection.amplitude
    imp_w = config.imperfection.width
    ng_local = N_G * (1.0 - imp_amp * np.exp(-x_coords**2 / (2.0 * imp_w**2)))
    inv_NG = Function(V0)
    inv_NG.vector().set_local(1.0 / np.maximum(ng_local, 1e-30))

    # ── BCs ────────────────────────────────────────────────────────────────
    u_bar_const = Constant(0.0)
    bc_left = DirichletBC(Vu, Constant(0.0), "near(x[0], -0.5)")
    bc_right = DirichletBC(Vu, u_bar_const, "near(x[0], 0.5)")
    bcs_u = [bc_left, bc_right]

    # ── UFL forms (use Function references; values update each iteration) ──
    u = TrialFunction(Vu)
    v = TestFunction(Vu)
    d_trial = TrialFunction(Vd)
    q = TestFunction(Vd)

    # Elasticity: ∫ (1-d_old)² u' v' dx = 0  (+ Dirichlet BC)
    a_u = (1.0 - d_old)**2 * u.dx(0) * v.dx(0) * dx
    L_u = Constant(0.0) * v * dx

    # Phase-field: ∫ [N_l² d'q' + (1 + 2N_l·inv_NG·H)d·q] dx = ∫ 2N_l·inv_NG·H·q dx
    a_d = (N_l**2 * d_trial.dx(0) * q.dx(0)
           + (1.0 + 2.0 * N_l * inv_NG * H_field) * d_trial * q) * dx
    L_d = (2.0 * N_l * inv_NG * H_field * q) * dx

    # ── Loading steps ──────────────────────────────────────────────────────
    du = config.loading.du
    u_final = config.loading.u_final
    u_steps = np.arange(du, u_final + 0.5 * du, du)
    tol = config.solver.atol
    max_iters = config.solver.max_iters

    # Storage (on sorted coordinates)
    sort_idx = np.argsort(x_coords)
    coords_sorted = x_coords[sort_idx]

    u_snaps, d_snaps, snap_times = [], [], []
    force_hist, u_bar_hist = [], []
    next_save = config.output.save_every

    # For convergence check
    eps_old_vals = np.zeros_like(x_coords)

    for step, u_bar in enumerate(u_steps):
        u_bar_const.assign(u_bar)

        # ── Staggered iteration ───────────────────────────────────────────
        for it in range(max_iters):
            # 1. Solve elasticity
            solve(a_u == L_u, u_new, bcs_u,
                  solver_parameters={"linear_solver": "default"})

            # 2. Compute strain and update history
            eps_new = project(u_new.dx(0), V0)
            eps_new_vals = eps_new.vector().get_local()
            H_vals = H_field.vector().get_local()
            H_new_vals = np.maximum(H_vals, 0.5 * eps_new_vals**2)
            H_field.vector().set_local(H_new_vals)

            # 3. Solve phase-field
            solve(a_d == L_d, d_new, [],
                  solver_parameters={"linear_solver": "default"})

            # 4. Check convergence
            err_u = np.max(np.abs(
                u_new.vector().get_local() - u_old.vector().get_local()
            ))
            err_d = np.max(np.abs(
                d_new.vector().get_local() - d_old.vector().get_local()
            ))
            err = max(err_u, err_d)

            u_old.assign(u_new)
            d_old.assign(d_new)

            if err < tol:
                break
        else:
            if step % 500 == 0:
                print(f"  Step {step}: stagger did NOT converge "
                      f"(err={err:.2e} after {max_iters} iters)")

        # ── Compute force (stress σ* = (1-d)² ε, constant along bar) ────
        d_vals = d_new.vector().get_local()
        eps_vals_final = project(u_new.dx(0), V0).vector().get_local()
        sigma_vals = (1.0 - d_vals)**2 * eps_vals_final
        force = np.mean(sigma_vals)

        force_hist.append(force)
        u_bar_hist.append(u_bar)

        # ── Save snapshots ───────────────────────────────────────────────
        if u_bar >= next_save - 0.5 * du or step == len(u_steps) - 1:
            u_snaps.append(u_new.vector().get_local()[sort_idx])
            d_snaps.append(d_new.vector().get_local()[sort_idx])
            snap_times.append(u_bar)
            next_save += config.output.save_every

        # Progress
        if step % 500 == 0:
            max_d = np.max(d_vals)
            print(f"  Step {step:5d}  u_bar={u_bar:.5f}  F={force:.6e}  "
                  f"max_d={max_d:.4f}  n_iter={it+1}")

    print(f"\nSolved {len(u_steps)} load steps.  "
          f"Final max_d = {np.max(d_new.vector().get_local()):.4f}")

    # ── Save results ─────────────────────────────────────────────────────────
    os.makedirs(config.output.dir, exist_ok=True)
    fpath = os.path.join(config.output.dir, config.output.filename)

    u_snaps_arr = np.array(u_snaps)
    d_snaps_arr = np.array(d_snaps)
    force_arr = np.array(force_hist)
    u_bar_arr = np.array(u_bar_hist)

    np.savez_compressed(fpath,
                        x=coords_sorted,
                        u_bar=u_bar_arr,
                        snap_u_bar=np.array(snap_times),
                        u=u_snaps_arr,
                        d=d_snaps_arr,
                        force=force_arr)
    print(f"Saved {fpath}")
    print(f"  u snaps: {u_snaps_arr.shape}, d snaps: {d_snaps_arr.shape}")
    print(f"  force-displacement: {force_arr.shape}")

    # ── Analytical comparison plots ──────────────────────────────────────────
    _plot_comparison(config, coords_sorted, u_bar_arr, force_arr,
                     u_snaps_arr, d_snaps_arr, snap_times,
                     N_l, N_G, alpha_nd, eps_c, sigma_c)

    # Cleanup
    del mesh, u_new, u_old, d_new, d_old, H_field, inv_NG
    gc.collect()


# ── Plotting ─────────────────────────────────────────────────────────────────

def _plot_comparison(config, x, u_bar, force, u_snaps, d_snaps, snap_times,
                     N_l, N_G, alpha, eps_c, sigma_c):
    """Generate 3-panel analytical comparison figure."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(config.output.plot_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # ── Panel 1: Force-displacement ───────────────────────────────────────
    ax = axes[0]
    ax.plot(u_bar, force, 'b-', linewidth=0.8, label='FEM (staggered)')

    # Analytical homogeneous solution
    eps_ana = np.linspace(1e-6, 0.1, 500)
    sigma_ana = analytical_homogeneous_stress(eps_ana, N_l, N_G)
    ax.plot(eps_ana, sigma_ana, 'r--', linewidth=1.5,
            label='Homogeneous (analytical)')

    ax.axvline(eps_c, color='gray', linestyle=':', linewidth=1,
               label=f'$\\varepsilon_c^* = {eps_c:.4f}$')
    ax.axhline(sigma_c, color='gray', linestyle=':', linewidth=1)
    ax.set_xlabel('$\\bar{u}^*$ (applied displacement)')
    ax.set_ylabel('$\\sigma^*$ (stress)')
    ax.set_title('Force–Displacement')
    ax.legend(fontsize=8)
    ax.set_xlim(0, u_bar[-1])
    ax.grid(True, alpha=0.3)

    # ── Panel 2: Crack profile at final step ──────────────────────────────
    ax = axes[1]
    d_final = d_snaps[-1]
    # Find crack centre (where d is max)
    xc_idx = np.argmax(d_final)
    xc = x[xc_idx]

    ax.plot(x, d_final, 'b-', linewidth=1.0, label=f'FEM (u_bar={snap_times[-1]:.4f})')
    ax.plot(x, analytical_crack_profile(x, xc, N_l), 'r--',
            linewidth=2.0, label=f'$\\exp(-|x-x_c|/N_\\ell)$')
    ax.set_xlabel('$x^*$')
    ax.set_ylabel('$d$ (damage)')
    ax.set_title(f'Crack Profile  ($x_c={xc:.4f}$, $N_\\ell={N_l:.4f}$)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Panel 3: Damage onset (d vs strain at centre) ─────────────────────
    ax = axes[2]
    # Extract centre damage and strain over loading
    xc_idx_snap = np.argmin(np.abs(x - xc))
    d_center = d_snaps[:, xc_idx_snap]
    # Approximate strain from homogeneous: ε ≈ u_bar / L = u_bar (domain = 1)
    eps_global = np.array(snap_times)  # u_bar ≈ global strain before localisation

    ax.plot(eps_global, d_center, 'b.-', markersize=2, linewidth=0.8,
            label='FEM (centre point)')

    # Analytical homogeneous
    eps_fine = np.linspace(0, eps_global[-1], 300)
    d_ana = analytical_homogeneous_damage(eps_fine, N_l, N_G)
    ax.plot(eps_fine, d_ana, 'r--', linewidth=1.5,
            label='Homogeneous (analytical)')
    ax.axvline(eps_c, color='gray', linestyle=':', linewidth=1,
               label=f'$\\varepsilon_c^* = {eps_c:.4f}$')

    ax.set_xlabel('$\\bar{u}^*$ (global strain)')
    ax.set_ylabel('$d$ (damage at centre)')
    ax.set_title('Damage Onset')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig_path = os.path.join(config.output.plot_dir, 'analytical_comparison.png')
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"Comparison plot saved to {fig_path}")


# ── Entry point ──────────────────────────────────────────────────────────────

_DEFAULT = os.path.join(os.path.dirname(__file__), 'configs', 'solver_base.py')
_CONFIG = _config_flags.DEFINE_config_file('config', _DEFAULT, 'Config file')
FLAGS = _absl_app.flags.FLAGS


def _main(_):
    solve_fracture(FLAGS.config)


if __name__ == '__main__':
    _absl_app.run(_main)
