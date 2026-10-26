"""2D FEM solver for KKS phase-field pitting corrosion (non-dimensional, MPI).

Environment:
  conda activate fenicsenv

Usage:
  python examples/corrosion2d/solver.py --config=examples/corrosion2d/configs/solver_1pit.py
  mpirun --allow-run-as-root -np 12 python examples/corrosion2d/solver.py \
      --config=examples/corrosion2d/configs/solver_1pit.py
"""

import gc, os, sys
import numpy as np

# First try the fenicsenv conda prefix, then sys.prefix
_fenics_prefix = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
# Use the conda environment prefix
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


# ── Initial conditions ──────────────────────────────────────────────────────

class _PitIC(UserExpression):
    """2D pit initial condition: φ=0 inside pit, φ=1 in solid."""

    def __init__(self, k, pits, c_se, **kwargs):
        super().__init__(**kwargs)
        self.k = k
        self.pits = pits  # list of (cx, cy, R)
        self.c_se = c_se

    def eval(self, value, x):
        phi = 1.0  # default: solid everywhere
        for cx, cy, R in self.pits:
            dist = np.sqrt((x[0] - cx)**2 + (x[1] - cy)**2)
            phi_pit = 0.5 * (1.0 + np.tanh(self.k * (dist - R)))
            phi = np.minimum(phi, phi_pit)
        c = _h(phi) * self.c_se
        value[0] = phi
        value[1] = c

    def value_shape(self):
        return (2,)


# ── Boundary marker: pit mouth (small exposed region) ───────────────────────

# ── MPI gather ──────────────────────────────────────────────────────────────

def _gather_2d(sol, V, comm):
    """Gather φ and c + coordinates to rank 0, deduplicate, sort into grid."""
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
        coords_list = [local_coords]
        phi_list = [local_phi]
        c_list = [local_c]

    if comm.rank == 0:
        all_coords = np.vstack(coords_list)
        all_phi = np.concatenate(phi_list)
        all_c = np.concatenate(c_list)
        # Deduplicate (ghost DOFs shared between processes)
        _, unique_idx = np.unique(all_coords.round(decimals=10),
                                   axis=0, return_index=True)
        # Sort by (y, x) for grid ordering
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
    ic = _PitIC(k=k, pits=list(config.pits.entries), c_se=phys.c_se, degree=1)
    sol.interpolate(ic)
    sol_prev.interpolate(ic)

    # ── split fields ──────────────────────────────────────────────────────
    phi, c = split(sol)
    phi_prev, c_prev = split(sol_prev)
    v_phi, v_c = TestFunctions(V)
    dt_const = Constant(config.time.dt)

    # ── BC: pit mouth = pit interior away from transition zone ──────────────
    pits = list(config.pits.entries)

    class _PitMouth(SubDomain):
        def inside(self, x, on_boundary):
            # Pointwise BC checks do not set on_boundary. Restrict to the
            # bottom edge explicitly, so a narrow mouth cannot miss all facets.
            if not near(x[1], 0.0):
                return False
            for cx, cy, R in pits:
                d = np.sqrt((x[0]-cx)**2 + (x[1]-cy)**2)
                if d < R * 0.5:
                    return True
            return False

    class _SolidBoundary(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and (near(x[0], 0.0) or near(x[0], 1.0)
                                    or near(x[1], 0.5))

    pit_mouth = _PitMouth()
    solid_boundary = _SolidBoundary()
    bcs = [
        DirichletBC(V.sub(0), Constant(0.0), pit_mouth, method="pointwise"),
        DirichletBC(V.sub(1), Constant(0.0), pit_mouth, method="pointwise"),
        DirichletBC(V.sub(0), Constant(1.0), solid_boundary),
        DirichletBC(V.sub(1), Constant(1.0), solid_boundary),
    ]
    for bc in bcs:
        bc.apply(sol.vector())
    sol_prev.assign(sol)

    # ── time integration method ───────────────────────────────────────────
    # theta = 1.0: backward Euler (default, O(dt))
    # theta = 0.5: Crank-Nicolson (O(dt²))
    theta = config.time.theta

    # ── weak form (scaled for conditioning) ───────────────────────────────
    scale_ch = 1.0 / max(N_CH, 1.0)
    scale_ac = 1.0 / max(N_AC2, 1.0)

    # midpoint values for theta method
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
    # prm['relaxation_parameter'] = 0.9

    # ── time stepping ─────────────────────────────────────────────────────
    dt_main = config.time.dt
    dt_init = dt_main / 100.0  # tiny dt for initial relaxation
    n_relax = 20               # number of relaxation steps
    save_every = config.output.save_every

    if rank == 0:
        print(f"nx={nx}, ny={ny}, dt*={dt_main}, "
              f"save every {save_every}")

    # Record IC
    coords, phi_ic, c_ic = _gather_2d(sol, V, comm)
    if rank == 0:
        phi_snaps = [phi_ic]
        c_snaps = [c_ic]
        t_snaps = [0.0]

    t_current = 0.0
    # ── relaxation phase ────────────────────────────────────────────────
    for step in range(1, n_relax + 1):
        t_current += dt_init
        dt_const.assign(dt_init)
        sol_prev.assign(sol)
        solver.solve()
        if rank == 0 and step % 5 == 0:
            print(f"  Relax {step}/{n_relax} t*={t_current:.6f}")

    # ── main time stepping ──────────────────────────────────────────────
    dt_const.assign(dt_main)
    next_save = save_every  # first save at t=save_every
    main_step = 0
    while t_current < config.time.t_final - 1e-12:
        main_step += 1
        t_next = min(t_current + dt_main, next_save, config.time.t_final)
        dt_const.assign(t_next - t_current)
        sol_prev.assign(sol)
        solver.solve()
        t_current = t_next

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

    # ── save grid output ──────────────────────────────────────────────────
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

        # ── quick plot ────────────────────────────────────────────────────
        _plot_snapshots(config.output.dir, xs, ys, t_snaps, phi_grid, c_grid)

    del solver, problem, sol, sol_prev, mesh, V, bcs, F, J
    gc.collect()
    comm.Barrier()


# ── Plotting ────────────────────────────────────────────────────────────────

def _plot_snapshots(output_dir, xs, ys, times, phi_grid, c_grid):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Only plot key time snapshots
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
        for j, i in enumerate(indices):
            ax = axes[0][j]
            ax.pcolormesh(xs, ys, grid[i], cmap='coolwarm',
                          vmin=0, vmax=1, shading='auto')
            ax.set_title(f'{label} t*={times[i]:.2f}')
            ax.set_aspect('equal')
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f'{label}_snapshots.png'), dpi=150)
        plt.close(fig)
        plt.close(fig)

    print(f"Saved snapshots to {output_dir}")


# ── Config ──────────────────────────────────────────────────────────────────

_DEFAULT = os.path.join(os.path.dirname(__file__), 'configs', 'solver_1pit.py')
_CONFIG = _config_flags.DEFINE_config_file('config', _DEFAULT, 'Config file')
FLAGS = _absl_app.flags.FLAGS


def _main(_):
    solve(FLAGS.config)

if __name__ == '__main__':
    _absl_app.run(_main)
