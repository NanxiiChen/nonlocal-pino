"""Nonlocal peridynamic solver for 1D KKS phase-field corrosion.

Solves the same PDEs as the PINN (using L_δ instead of Δ), producing a
reference solution for direct comparison with both FEM and PINN results.

Uses a semi-implicit splitting: CH equation is linear in c (solved directly),
AC equation is nonlinear in φ (solved with scipy.optimize.fsolve).

Usage:
  python examples/corrosion1d/nonlocal_solver.py
  python examples/corrosion1d/nonlocal_solver.py --config=examples/corrosion1d/configs/solver_diffusion.py
"""

import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import jax, jax.numpy as jnp
jax.config.update("jax_default_matmul_precision", "highest")
import numpy as np
from scipy.optimize import fsolve
from pinn.operators import build_D

from absl import app, flags
from ml_collections import config_flags

_DEFAULT = os.path.join(os.path.dirname(__file__), 'configs', 'solver_diffusion.py')
_CONFIG = config_flags.DEFINE_config_file('config', _DEFAULT, 'Config file')
FLAGS = flags.FLAGS


# ── Phase-field helpers ──────────────────────────────────────────────────────

def _h(phi):
    return -2.0 * phi**3 + 3.0 * phi**2

def _hp(phi):
    return -6.0 * phi**2 + 6.0 * phi

def _gp(phi):
    return 4.0 * phi**3 - 6.0 * phi**2 + 2.0 * phi

# ── Main solver ──────────────────────────────────────────────────────────────

def solve(config):
    phys = config.physical
    L_c = config.nondim.L_c
    T_c = config.nondim.T_c
    dc0 = phys.c_se - phys.c_le

    N_CH  = phys.D * T_c / L_c**2
    N_AC1 = 2.0 * phys.A * phys.L * T_c
    N_AC2 = phys.L * phys.omega_phi * T_c
    N_AC3 = phys.L * phys.alpha_phi * T_c / L_c**2

    print(f"N_CH={N_CH:.4e}  N_AC1={N_AC1:.4e}  N_AC2={N_AC2:.4e}  N_AC3={N_AC3:.4e}")
    label = "diffusion-controlled" if phys.L >= 1.0 else "activation-controlled"
    print(f"L={phys.L:.2e}  ({label})")

    # ── Spatial discretization ────────────────────────────────────────────
    nx = config.mesh.n_cells + 1
    x_all = np.linspace(0.0, 1.0, nx)
    dx = x_all[1] - x_all[0]

    delta = config.nl.delta
    kernel = config.nl.kernel
    D_np = np.array(build_D(jnp.array(x_all)[:, None], delta, kernel=kernel))
    print(f"D matrix: {D_np.shape[0]}x{D_np.shape[1]}, δ={delta}, kernel={kernel}, dx={dx:.4f}")

    # ── Initial condition ─────────────────────────────────────────────────
    k = np.sqrt(phys.omega_phi / (2.0 * phys.alpha_phi)) * L_c
    center = config.init.center
    dist = x_all - center
    phi_n = 0.5 * (1.0 - np.tanh(k * dist))
    c_n = _h(phi_n) * phys.c_se

    dt = config.time.dt
    n_steps = int(config.time.t_final / dt)

    # ── Build free/constr: only endpoints constrained ─────────────────────
    # Peridynamics handles boundary truncation naturally via the horizon.
    mask_free = np.ones(nx, dtype=bool)
    mask_free[0] = False
    mask_free[-1] = False
    free_idx = np.where(mask_free)[0]
    constr_idx = np.where(~mask_free)[0]
    n_free = len(free_idx)

    # BC values at constrained nodes
    phi_bc = np.zeros(nx)
    phi_bc[0] = 1.0
    c_bc = np.zeros(nx)
    c_bc[0] = phys.c_se

    # Pre-build linear operators for free DOFs
    D_ff = D_np[free_idx][:, free_idx]
    D_fc = D_np[free_idx][:, constr_idx]
    I_ff = np.eye(n_free)
    A_ch = I_ff / dt + N_CH * D_ff  # for CH linear solve

    phi_hist = [phi_n.copy()]
    c_hist = [c_n.copy()]
    t_hist = [0.0]

    print(f"Time stepping: dt={dt}, steps={n_steps}, "
          f"free={n_free} constr={nx-n_free} (δ/Δx≈{delta/dx:.1f})")
    t_start = time.perf_counter()

    for step in range(n_steps):
        # Picard iteration: alternate CH (linear) and AC (nonlinear) within step
        phi_star = phi_n.copy()
        c_star = c_n.copy()
        phi_star_free = phi_n[free_idx].copy()
        c_star_free = c_n[free_idx].copy()

        for picard_iter in range(5):
            phi_star_prev = phi_star_free.copy()
            c_star_prev = c_star_free.copy()

            # ── CH: solve linear system for c* given φ* ──────────────────
            h_full = _h(phi_star)
            b_c = c_n[free_idx] / dt \
                + N_CH * dc0 * (D_np @ h_full)[free_idx] \
                - N_CH * D_fc @ c_star[constr_idx]
            c_star_free = np.linalg.solve(A_ch, b_c)
            c_star[free_idx] = c_star_free
            c_star[0] = phys.c_se; c_star[-1] = 0.0

            # ── AC: nonlinear solve for φ* given c* ──────────────────────
            c_full = c_star.copy()
            phi_bc_full = phi_bc.copy()

            def ac_residual(phi_free):
                phi = phi_bc_full.copy()
                phi[free_idx] = phi_free
                D_phi = D_np @ phi
                F = (phi_free - phi_n[free_idx]) / dt \
                    + N_AC3 * D_phi[free_idx] \
                    + N_AC2 * _gp(phi_free) \
                    - N_AC1 * (c_full[free_idx] - _h(phi_free) * dc0 - phys.c_le) \
                           * dc0 * _hp(phi_free)
                return F

            phi_new_free = fsolve(
                ac_residual, phi_star_free.copy(),
                maxfev=1000, xtol=1e-12,
            )
            phi_star_free = phi_new_free
            phi_star[free_idx] = phi_star_free
            phi_star[0] = 1.0; phi_star[-1] = 0.0

            # Check Picard convergence
            delta_phi = np.max(np.abs(phi_star_free - phi_star_prev))
            c_scale = np.max(np.abs(c_star_prev))
            delta_c = np.max(np.abs(c_star_free - c_star_prev))
            delta_c /= c_scale if c_scale > 1e-12 else 1.0
            if picard_iter >= 2 and delta_phi < 1e-10 and delta_c < 1e-10:
                break

        phi_n = phi_star.copy()
        c_n = c_star.copy()

        phi_hist.append(phi_n.copy())
        c_hist.append(c_n.copy())
        t_hist.append((step + 1) * dt)

        if (step + 1) % 20 == 0 or step == 0:
            elapsed = time.perf_counter() - t_start
            print(f"  Step {step+1}/{n_steps}  t*={(step+1)*dt:.3f}  ({elapsed:.1f}s)")

    # ── Save ──────────────────────────────────────────────────────────────
    output_dir = config.output.dir
    os.makedirs(output_dir, exist_ok=True)
    base = config.output.filename.replace('.npz', '')
    fpath = os.path.join(output_dir, f'{base}_nonlocal.npz')
    np.savez_compressed(fpath,
                        mesh=x_all,
                        time=np.array(t_hist),
                        phi=np.array(phi_hist),
                        c=np.array(c_hist))
    print(f"Saved {fpath}  (phi shape: {np.array(phi_hist).shape})")

    # ── Compare with FEM ──────────────────────────────────────────────────
    fem_path = os.path.join(output_dir, config.output.filename)
    if os.path.exists(fem_path):
        _plot_comparison(x_all, np.array(t_hist),
                         np.array(phi_hist), np.array(c_hist),
                         fem_path, output_dir, base, delta)
    else:
        print(f"FEM data not found at {fem_path}, skipping comparison plot")

    elapsed = time.perf_counter() - t_start
    print(f"Total time: {elapsed:.1f}s")


# ── Comparison plots ─────────────────────────────────────────────────────────

def _plot_comparison(x, t, phi_nl, c_nl, fem_path, output_dir, base, delta):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fem = np.load(fem_path)
    x_fem = fem['mesh']
    t_fem = fem['time']
    phi_fem = fem['phi']
    c_fem = fem['c']

    from scipy.interpolate import RegularGridInterpolator
    phi_interp = RegularGridInterpolator((t, x), phi_nl, bounds_error=False,
                                          fill_value=None)
    c_interp = RegularGridInterpolator((t, x), c_nl, bounds_error=False,
                                        fill_value=None)
    TT, XX = np.meshgrid(t_fem, x_fem, indexing='ij')
    pts = np.stack([TT.ravel(), XX.ravel()], axis=-1)
    phi_on_fem = phi_interp(pts).reshape(TT.shape)
    c_on_fem = c_interp(pts).reshape(TT.shape)

    dphi = phi_on_fem - phi_fem
    dc = c_on_fem - c_fem

    rel_phi = np.sqrt(np.mean(dphi**2)) / max(np.sqrt(np.mean(phi_fem**2)), 1e-16)
    rel_c = np.sqrt(np.mean(dc**2)) / max(np.sqrt(np.mean(c_fem**2)), 1e-16)

    extent = [float(x_fem[0]), float(x_fem[-1]), float(t_fem[0]), float(t_fem[-1])]

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))

    for row, (nl, fem_data, diff, label, rel_err) in enumerate([
        (phi_on_fem, phi_fem, dphi, r'$\phi$', rel_phi),
        (c_on_fem, c_fem, dc, '$c$', rel_c)]):

        im = axes[row, 0].imshow(nl, cmap='coolwarm', vmin=0, vmax=1,
                                  aspect='auto', origin='lower', extent=extent)
        axes[row, 0].set_title(f'{label} nonlocal (δ={delta})')
        plt.colorbar(im, ax=axes[row, 0])

        im = axes[row, 1].imshow(fem_data, cmap='coolwarm', vmin=0, vmax=1,
                                  aspect='auto', origin='lower', extent=extent)
        axes[row, 1].set_title(f'{label} FEM (local)')
        plt.colorbar(im, ax=axes[row, 1])

        vmax = max(abs(float(diff.min())), abs(float(diff.max())), 1e-6)
        im = axes[row, 2].imshow(diff, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                                  aspect='auto', origin='lower', extent=extent)
        axes[row, 2].set_title(f'{label} diff  (rel L2 = {rel_err:.3f})')
        plt.colorbar(im, ax=axes[row, 2])

    for ax in axes.flat:
        ax.set_xlabel(r'$x^*$'); ax.set_ylabel(r'$t^*$')

    fig.suptitle(f'Nonlocal (δ={delta}, {x.shape[0]} pts) vs FEM ({x_fem.shape[0]} pts)',
                 fontsize=13)
    fig.tight_layout()
    plot_path = os.path.join(output_dir, f'{base}_nonlocal_comparison.png')
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Saved comparison plot: {plot_path}")


# ── Entry point ──────────────────────────────────────────────────────────────

def _main(_):
    solve(FLAGS.config)

if __name__ == '__main__':
    app.run(_main)
