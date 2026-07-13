"""2D FEM solver for KKS electropolishing — rough interface (non-dimensional, MPI).

Environment:
  conda activate fenicsenv

Usage:
  python examples/electropolish2d/solver.py --config=examples/electropolish2d/configs/solver_base.py
  mpirun --allow-run-as-root -np 18 python examples/electropolish2d/solver.py \
      --config=examples/electropolish2d/configs/solver_base.py
"""

import gc, os, sys
import numpy as np

_pkgconfig_dir = '/root/miniconda3/envs/fenicsenv/lib/pkgconfig'
if not os.path.isdir(_pkgconfig_dir):
    _pkgconfig_dir = os.path.join(sys.prefix, 'lib', 'pkgconfig')
if os.path.isdir(_pkgconfig_dir):
    os.environ.setdefault('PKG_CONFIG_PATH', '')
    if _pkgconfig_dir not in os.environ['PKG_CONFIG_PATH']:
        os.environ['PKG_CONFIG_PATH'] = _pkgconfig_dir + ':' + os.environ['PKG_CONFIG_PATH']

from absl import app as _absl_app
from ml_collections import config_flags as _config_flags
from mpi4py import MPI as _MPI
from fenics import (
    Constant, DirichletBC, FiniteElement, Function, FunctionSpace,
    RectangleMesh, Point, NonlinearVariationalProblem,
    NonlinearVariationalSolver, SubDomain, TestFunctions, TrialFunction,
    UserExpression, derivative, dot, dx, grad, near, split,
)


# ── UFL helpers ─────────────────────────────────────────────────────────────

def _h(phi):
    return -2.0 * phi**3 + 3.0 * phi**2

def _hp(phi):
    return -6.0 * phi**2 + 6.0 * phi

def _gp(phi):
    return 4.0 * phi**3 - 6.0 * phi**2 + 2.0 * phi


# ── Initial condition — rough interface ────────────────────────────────────

def _rough_surface(x, y0, modes, rng):
    """Evaluate rough surface height at x coordinates."""
    y = np.full_like(x, y0)
    for amp, freq, phase in modes:
        y += amp * (np.sin(2.0 * np.pi * freq * x + phase)
                    + np.cos(2.0 * np.pi * freq * x + phase + 0.3))
    return y


class _RoughIC(UserExpression):
    """φ = tanh(k * sdf(x)), bilinearly interpolated from precomputed grid."""

    def __init__(self, k, y0, modes, c_se, **kwargs):
        super().__init__(**kwargs)
        # Precompute SDF on a dense grid, then evaluate phi once
        _nx, _ny = 2000, 1000
        self._xs = np.linspace(0, 1, _nx)
        self._ys = np.linspace(0, 0.5, _ny)
        x_curve = np.linspace(0, 1, _nx)
        y_curve = np.full_like(x_curve, y0)
        for amp, freq, phase in modes:
            y_curve += amp * (np.sin(2.0 * np.pi * freq * x_curve + phase)
                              + np.cos(2.0 * np.pi * freq * x_curve + phase + 0.3))
        Yg, Xg = np.meshgrid(self._ys, self._xs, indexing='ij')
        sdf = _segment_sdf(Xg.ravel(), Yg.ravel(), x_curve, y_curve,
                           _rough_surface(Xg.ravel(), y0, modes, None)).reshape(_ny, _nx)
        self._phi = 0.5 * (1.0 + np.tanh(k * sdf))
        self._c = _h(self._phi) * c_se
        self._dx = self._xs[1] - self._xs[0]
        self._dy = self._ys[1] - self._ys[0]

    def eval(self, value, x):
        ix = (x[0] - self._xs[0]) / self._dx
        iy = (x[1] - self._ys[0]) / self._dy
        ix = int(np.clip(ix, 0, len(self._xs) - 1))
        iy = int(np.clip(iy, 0, len(self._ys) - 1))
        value[0] = self._phi[iy, ix]
        value[1] = self._c[iy, ix]

    def value_shape(self):
        return (2,)


def _segment_sdf(xq, yq, x_curve, y_curve, y_surf_at_xq, window=16):
    """Signed distance to piecewise-linear profile.  + below, - above."""
    s0x = x_curve[:-1]; s0y = y_curve[:-1]
    dx = x_curve[1:] - s0x; dy = y_curve[1:] - s0y
    len2 = dx*dx + dy*dy + 1e-20
    n_seg, n_q = len(s0x), len(xq)
    xi_idx = np.searchsorted(x_curve, xq)
    dist2 = np.full(n_q, np.inf)
    for offset in range(-window, window + 1):
        seg = np.clip(xi_idx + offset, 0, n_seg - 1)
        sx, sdx, sy, sdy, sl2 = s0x[seg], dx[seg], s0y[seg], dy[seg], len2[seg]
        rx, ry = xq - sx, yq - sy
        t = np.clip((rx*sdx + ry*sdy) / sl2, 0.0, 1.0)
        px, py = sx + t*sdx, sy + t*sdy
        d2 = (xq - px)**2 + (yq - py)**2
        dist2 = np.minimum(dist2, d2)
    return np.where(yq < y_surf_at_xq, 1.0, -1.0) * np.sqrt(dist2)


# ── MPI gather ─────────────────────────────────────────────────────────────

def _gather_2d(sol, V, comm):
    phi_f, c_f = sol.split(deepcopy=True)
    V_phi = V.sub(0).collapse()
    local_coords = V_phi.tabulate_dof_coordinates().reshape(-1, 2)
    local_phi = phi_f.vector().get_local()
    local_c = c_f.vector().get_local()

    if comm.size > 1:
        coords_list = comm.gather(local_coords, root=0)
        phi_list = comm.gather(local_phi, root=0)
        c_list = comm.gather(local_c, root=0)
    else:
        coords_list, phi_list, c_list = [local_coords], [local_phi], [local_c]

    if comm.rank == 0:
        all_coords = np.vstack(coords_list)
        all_phi = np.concatenate(phi_list)
        all_c = np.concatenate(c_list)
        _, unique_idx = np.unique(all_coords.round(decimals=10),
                                   axis=0, return_index=True)
        uc = all_coords[unique_idx]
        sort_idx = np.lexsort((uc[:, 0], uc[:, 1]))
        return (all_coords[unique_idx][sort_idx],
                all_phi[unique_idx][sort_idx],
                all_c[unique_idx][sort_idx])
    return None, None, None


# ── Solver ──────────────────────────────────────────────────────────────────

def solve(config):
    comm = _MPI.COMM_WORLD
    rank = comm.rank

    phys = config.physical
    L_c = config.nondim.L_c
    T_c = config.nondim.T_c
    dc0 = phys.c_se - phys.c_le

    N_CH  = phys.D * T_c / L_c**2
    N_AC1 = 2.0 * phys.A * phys.L * T_c
    N_AC2 = phys.L * phys.omega_phi * T_c
    N_AC3 = phys.L * phys.alpha_phi * T_c / L_c**2

    if rank == 0:
        print(f"N_CH={N_CH:.4e} N_AC1={N_AC1:.4e} "
              f"N_AC2={N_AC2:.4e} N_AC3={N_AC3:.4e}")

    # ── mesh ──────────────────────────────────────────────────────────────
    nx, ny = config.mesh.nx, config.mesh.ny
    mesh = RectangleMesh(comm, Point(0.0, 0.0), Point(1.0, 0.5), nx, ny)
    P1 = FiniteElement('CG', mesh.ufl_cell(), 1)
    V = FunctionSpace(mesh, P1 * P1)

    sol = Function(V)
    sol_prev = Function(V)

    # ── IC ────────────────────────────────────────────────────────────────
    k = np.sqrt(phys.omega_phi / (2.0 * phys.alpha_phi)) * L_c
    rng = np.random.RandomState(config.surface.seed)
    modes = []
    for i in range(config.surface.n_modes):
        amp = rng.uniform(0.3, 1.0) * config.surface.amplitude
        freq = rng.uniform(config.surface.freq_lo, config.surface.freq_hi)
        phase = rng.uniform(0.0, 2.0 * np.pi)
        modes.append((amp, freq, phase))

    ic = _RoughIC(k=k, y0=config.surface.y0, modes=modes,
                  c_se=phys.c_se, degree=1)
    sol.interpolate(ic)
    sol_prev.interpolate(ic)

    # ── split fields ──────────────────────────────────────────────────────
    phi, c = split(sol)
    phi_prev, c_prev = split(sol_prev)
    v_phi, v_c = TestFunctions(V)
    dt_const = Constant(config.time.dt)

    # ── BC ────────────────────────────────────────────────────────────────
    # Top (y=0.5): liquid — φ=0, c=0
    # Bottom (y=0): solid — φ=1, c=1
    # Left/Right: Neumann (natural in FEniCS — no BC objects)
    class _Top(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and near(x[1], 0.5)

    class _Bottom(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and near(x[1], 0.0)

    top, bottom = _Top(), _Bottom()
    bcs = [
        DirichletBC(V.sub(0), Constant(0.0), top),
        DirichletBC(V.sub(1), Constant(0.0), top),
        DirichletBC(V.sub(0), Constant(1.0), bottom),
        DirichletBC(V.sub(1), Constant(1.0), bottom),
    ]

    # ── weak form ────────────────────────────────────────────────────────
    theta = config.time.theta
    scale_ch = 1.0 / max(N_CH, 1.0)
    scale_ac = 1.0 / max(N_AC2, 1.0)

    c_th = theta * c + (1.0 - theta) * c_prev
    phi_th = theta * phi + (1.0 - theta) * phi_prev

    F_c = (c - c_prev) / dt_const * v_c * dx \
        + N_CH * dot(grad(c_th), grad(v_c)) * dx \
        - N_CH * dc0 * dot(grad(_h(phi_th)), grad(v_c)) * dx

    F_phi = (phi - phi_prev) / dt_const * v_phi * dx \
        + N_AC3 * dot(grad(phi_th), grad(v_phi)) * dx \
        + N_AC2 * _gp(phi_th) * v_phi * dx \
        - N_AC1 * (c_th - _h(phi_th) * dc0 - phys.c_le) * dc0 * _hp(phi_th) * v_phi * dx

    F = scale_ch * F_c + scale_ac * F_phi
    J = derivative(F, sol, TrialFunction(V))

    # ── solver ────────────────────────────────────────────────────────────
    problem = NonlinearVariationalProblem(F, sol, bcs, J)
    solver = NonlinearVariationalSolver(problem)
    prm = solver.parameters['newton_solver']
    prm['absolute_tolerance'] = config.solver.atol
    prm['relative_tolerance'] = config.solver.rtol
    prm['maximum_iterations'] = config.solver.max_iters
    prm['convergence_criterion'] = 'incremental'
    prm['linear_solver'] = 'mumps'

    # ── time stepping ─────────────────────────────────────────────────────
    dt_main = config.time.dt
    dt_init = dt_main / 50.0
    n_relax = 10
    save_every = config.output.save_every

    if rank == 0:
        print(f"nx={nx}, ny={ny}, dt*={dt_main}, "
              f"save every {save_every}")

    coords, phi_ic, c_ic = _gather_2d(sol, V, comm)
    if rank == 0:
        phi_snaps, c_snaps, t_snaps = [phi_ic], [c_ic], [0.0]

    t_current = 0.0
    # ── relaxation ────────────────────────────────────────────────────────
    for step in range(1, n_relax + 1):
        t_current += dt_init
        dt_const.assign(dt_init)
        sol_prev.assign(sol)
        solver.solve()
        if rank == 0 and step % 5 == 0:
            print(f"  Relax {step}/{n_relax} t*={t_current:.6f}")

    # ── main loop ─────────────────────────────────────────────────────────
    dt_const.assign(dt_main)
    next_save = save_every
    main_step = 0
    while t_current < config.time.t_final - 1e-12:
        main_step += 1
        t_current += dt_main
        t_current = min(t_current, config.time.t_final)
        sol_prev.assign(sol)
        solver.solve()

        if t_current >= next_save - 1e-12 or t_current >= config.time.t_final - 1e-12:
            coords, phi_vals, c_vals = _gather_2d(sol, V, comm)
            if rank == 0:
                phi_snaps.append(phi_vals)
                c_snaps.append(c_vals)
                t_snaps.append(t_current)
                print(f"  Step {main_step} t*={t_current:.4f} [saved]")
            next_save += save_every
        elif rank == 0 and main_step % 50 == 0:
            print(f"  Step {main_step} t*={t_current:.4f}")

    # ── save ──────────────────────────────────────────────────────────────
    if rank == 0:
        os.makedirs(config.output.dir, exist_ok=True)
        xs = np.linspace(0.0, 1.0, nx + 1)
        ys = np.linspace(0.0, 0.5, ny + 1)
        X, Y = np.meshgrid(xs, ys, indexing='xy')

        n_verts = (nx + 1) * (ny + 1)
        phi_grid = np.array([p[:n_verts].reshape(ny+1, nx+1) for p in phi_snaps])
        c_grid = np.array([c[:n_verts].reshape(ny+1, nx+1) for c in c_snaps])

        fpath = os.path.join(config.output.dir, config.output.filename)
        np.savez_compressed(fpath,
                            X=X, Y=Y,
                            time=np.array(t_snaps),
                            phi=phi_grid, c=c_grid)
        print(f"Saved {fpath}  (snapshots={len(t_snaps)}, "
              f"phi shape={phi_grid.shape})")

        _plot_snapshots(config.output.dir, xs, ys, t_snaps, phi_grid, c_grid)

    del solver, problem, sol, sol_prev, mesh, V, bcs, F, J
    gc.collect()
    comm.Barrier()


# ── Plotting ────────────────────────────────────────────────────────────────

def _plot_snapshots(output_dir, xs, ys, times, phi_grid, c_grid):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    target_times = [0.0, 0.25, 0.5, 0.75, 1.0]
    indices = []
    for tt in target_times:
        idx = int(np.argmin(np.abs(np.array(times) - tt)))
        indices.append(idx)
    indices = sorted(set(indices))

    for label, grid in [('phi', phi_grid), ('c', c_grid)]:
        fig, axes = plt.subplots(1, len(indices),
                                  figsize=(3*len(indices), 3),
                                  squeeze=False)
        vmin = 0.0 if label == 'phi' else None
        for ax_j, idx in enumerate(indices):
            im = axes[0][ax_j].imshow(grid[idx],
                                       extent=[xs[0], xs[-1], ys[0], ys[-1]],
                                       origin='lower', cmap='coolwarm',
                                       vmin=vmin, vmax=1.0,
                                       interpolation='bilinear')
            axes[0][ax_j].set_title(f'{label}  t*={times[idx]:.2f}')
            axes[0][ax_j].set_aspect('equal')
        plt.colorbar(im, ax=axes[0][-1])
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f'snapshot_{label}.png'), dpi=100)
        plt.close(fig)

    print(f"Plots saved to {output_dir}")


# ── Config / entry ──────────────────────────────────────────────────────────

_DEFAULT = os.path.join(os.path.dirname(__file__), 'configs', 'solver_base.py')
_CONFIG = _config_flags.DEFINE_config_file('config', _DEFAULT, 'Config file')
FLAGS = _absl_app.flags.FLAGS


def _main(_):
    solve(FLAGS.config)


if __name__ == '__main__':
    _absl_app.run(_main)
