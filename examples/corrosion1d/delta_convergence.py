"""δ-convergence: nonlocal solver vs FEM for a range of δ.

Usage:
  python examples/corrosion1d/delta_convergence.py --config=examples/corrosion1d/configs/solver_activation.py
  python examples/corrosion1d/delta_convergence.py --config=examples/corrosion1d/configs/solver_diffusion.py
"""

import os, sys, time, csv
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import jax, jax.numpy as jnp
jax.config.update("jax_default_matmul_precision", "highest")
import numpy as np
from scipy.optimize import fsolve
from pinn.operators import build_D

from absl import app, flags as _abs_flags
from ml_collections import config_flags

_DEFAULT = os.path.join(os.path.dirname(__file__), 'configs', 'solver_diffusion.py')
_CONFIG = config_flags.DEFINE_config_file('config', _DEFAULT, 'Config file')
FLAGS = _abs_flags.FLAGS


def _h(phi): return -2.0 * phi**3 + 3.0 * phi**2
def _hp(phi): return -6.0 * phi**2 + 6.0 * phi
def _gp(phi): return 4.0 * phi**3 - 6.0 * phi**2 + 2.0 * phi


def solve_one(phys, L_c, T_c, dc0, dt, t_final, center, delta, nx):
    N_CH  = phys.D * T_c / L_c**2
    N_AC1 = 2.0 * phys.A * phys.L * T_c
    N_AC2 = phys.L * phys.omega_phi * T_c
    N_AC3 = phys.L * phys.alpha_phi * T_c / L_c**2

    x_all = np.linspace(0.0, 1.0, nx)
    D_np = np.array(build_D(jnp.array(x_all)[:, None], delta, kernel="cubic"))

    k = np.sqrt(phys.omega_phi / (2.0 * phys.alpha_phi)) * L_c
    phi_n = 0.5 * (1.0 - np.tanh(k * (x_all - center)))
    c_n = _h(phi_n) * phys.c_se

    n_steps = int(t_final / dt)
    mask_free = np.ones(nx, dtype=bool)
    mask_free[0] = False; mask_free[-1] = False
    free_idx = np.where(mask_free)[0]
    constr_idx = np.where(~mask_free)[0]
    n_free = len(free_idx)

    phi_bc = np.zeros(nx); phi_bc[0] = 1.0
    D_ff = D_np[free_idx][:, free_idx]
    D_fc = D_np[free_idx][:, constr_idx]
    A_ch = np.eye(n_free) / dt + N_CH * D_ff

    phi_hist = [phi_n.copy()]
    for step in range(n_steps):
        phi_s, c_s = phi_n.copy(), c_n.copy()
        pf, cf = phi_n[free_idx].copy(), c_n[free_idx].copy()

        for _ in range(5):
            pp, cp = pf.copy(), cf.copy()

            hf = _h(phi_s)
            b_c = c_n[free_idx] / dt \
                + N_CH * dc0 * (D_np @ hf)[free_idx] \
                - N_CH * D_fc @ c_s[constr_idx]
            cf = np.linalg.solve(A_ch, b_c)
            c_s[free_idx] = cf; c_s[0] = phys.c_se; c_s[-1] = 0.0

            def ac_res(pf_in):
                ph = phi_bc.copy(); ph[free_idx] = pf_in
                return (pf_in - phi_n[free_idx]) / dt \
                    + N_AC3 * (D_np @ ph)[free_idx] \
                    + N_AC2 * _gp(pf_in) \
                    - N_AC1 * (c_s[free_idx] - _h(pf_in) * dc0 - phys.c_le) \
                           * dc0 * _hp(pf_in)

            pf = fsolve(ac_res, pf.copy(), maxfev=1000, xtol=1e-12)
            phi_s[free_idx] = pf; phi_s[0] = 1.0; phi_s[-1] = 0.0

            if np.max(np.abs(pf - pp)) < 1e-10:
                break

        phi_n = phi_s.copy(); c_n = c_s.copy()
        phi_hist.append(phi_n.copy())

    return np.array(phi_hist), x_all


def _main(_):
    config = FLAGS.config
    phys = config.physical
    L_c = config.nondim.L_c
    T_c = config.nondim.T_c
    dc0 = phys.c_se - phys.c_le
    base = config.output.filename.replace('.npz', '')

    fem_path = os.path.join(config.output.dir, config.output.filename)
    fem = np.load(fem_path)
    x_f, t_f, phi_f = fem['mesh'], fem['time'], fem['phi']

    deltas = [0.05, 0.02, 0.01, 0.007, 0.005]
    results = []
    dt = config.time.dt

    for delta in deltas:
        nx = int(1.0 / (delta / 5.0)) + 1  # δ/dx ≈ 5
        nx = min(nx, 501)  # cap for speed
        dx = 1.0 / (nx - 1)

        label = f'δ={delta:.3f} N={nx}'
        print(f"\n{'='*60}\n{label}\n{'='*60}")
        t0 = time.perf_counter()

        phi_nl, x_n = solve_one(phys, L_c, T_c, dc0, dt,
                                config.time.t_final, config.init.center,
                                delta, nx)

        from scipy.interpolate import RegularGridInterpolator
        t_n = np.linspace(0.0, config.time.t_final, phi_nl.shape[0])
        interp = RegularGridInterpolator((t_n, x_n), phi_nl, bounds_error=False)
        TT, XX = np.meshgrid(t_f, x_f, indexing='ij')
        phi_on = interp(np.stack([TT.ravel(), XX.ravel()], axis=-1)).reshape(TT.shape)
        rel = np.sqrt(np.mean((phi_on - phi_f)**2)) / max(np.sqrt(np.mean(phi_f**2)), 1e-16)

        results.append((delta, nx, dx, rel, x_n, phi_nl[-1]))
        print(f"  rel L2 φ = {rel:.4f}  ({time.perf_counter()-t0:.1f}s)")

    # Save CSV
    csv_path = os.path.join(config.output.dir, f'{base}_delta_conv.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['delta', 'nx', 'dx', 'rel_L2_phi'])
        for r in results:
            w.writerow(r[:4])
    print(f"\nSaved {csv_path}")

    # ── Plot ──────────────────────────────────────────────────────────────
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

    ds = [r[0] for r in results]
    rels = [r[3] for r in results]

    fig, (ax_err, ax_prof) = plt.subplots(1, 2, figsize=(14, 5))

    ax_err.loglog(ds, rels, 'o-', markersize=8, linewidth=2)
    ax_err.set_xlabel(r'$\delta$'); ax_err.set_ylabel(r'rel L2 error in $\phi$')
    ax_err.set_title(f'{base}: nonlocal → FEM convergence')
    ax_err.grid(True, alpha=0.3)

    ref = rels[-1] * (np.array(ds) / ds[-1])**2
    ax_err.loglog(ds, ref, 'k--', alpha=0.4, label=r'$O(\delta^2)$')
    ax_err.legend()

    for d, n, dx, r in results:
        ax_err.annotate(f'δ={d:.3f}\nN={n}', (d*1.05, r),
                        fontsize=7, va='center')

    # φ profiles at t_final
    from matplotlib.cm import viridis
    colors = viridis(np.linspace(0, 1, len(results) + 1))
    for i, r in enumerate(results):
        delta, nx, dx, rel, x_n, phi_last = r
        ax_prof.plot(x_n, phi_last, color=colors[i], alpha=0.7,
                     label=f'δ={delta:.3f} N={nx} err={rel:.4f}')
    ax_prof.plot(x_f, phi_f[-1], 'k-', linewidth=2, label='FEM (local)')
    ax_prof.set_xlabel(r'$x^*$'); ax_prof.set_ylabel(r'$\phi$')
    ax_prof.set_title(f'{base}: φ at t=1.0')
    ax_prof.legend(fontsize=7)
    ax_prof.set_xlim(0, 1)

    fig.tight_layout()
    plot_path = os.path.join(config.output.dir, f'{base}_delta_convergence.png')
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Saved {plot_path}")

    print(f"\n{'δ':>8s} {'N':>5s} {'δ/dx':>6s} {'rel L2 φ':>10s}")
    print("-" * 35)
    for r in results:
        d, n, dx, rel = r[:4]
        print(f"{d:8.3f} {n:5d} {d/dx:6.1f} {rel:10.4f}")


if __name__ == '__main__':
    app.run(_main)
