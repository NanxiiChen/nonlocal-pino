"""FEM solver for 1D phase-field pitting corrosion (non-dimensional, MPI-ready).

Environment:
  conda activate fenicsenv

Usage (run from project root):
  python examples/corrosion1d/fem.py
  python examples/corrosion1d/fem.py --config=my_config.py
  python examples/corrosion1d/fem.py --config.mesh.n_cells=200
  mpirun --allow-run-as-root -np 12 python examples/corrosion1d/solver.py --config=examples/corrosion1d/configs/solver_activation.py
"""

import gc
import os
import sys

# Ensure DOLFIN can find its pkg-config file (no manual PKG_CONFIG_PATH needed).
_pkgconfig_dir = os.path.join(sys.prefix, 'lib', 'pkgconfig')
if os.path.isdir(_pkgconfig_dir):
    os.environ.setdefault('PKG_CONFIG_PATH', '')
    if _pkgconfig_dir not in os.environ['PKG_CONFIG_PATH']:
        os.environ['PKG_CONFIG_PATH'] = _pkgconfig_dir + ':' + os.environ['PKG_CONFIG_PATH']

import numpy as np
from absl import app as _absl_app
from ml_collections import config_flags as _config_flags
from mpi4py import MPI as _MPI
from fenics import (
    Constant,
    DirichletBC,
    FiniteElement,
    Function,
    FunctionSpace,
    IntervalMesh,
    NonlinearVariationalProblem,
    NonlinearVariationalSolver,
    TestFunctions,
    TrialFunction,
    UserExpression,
    derivative,
    dot,
    dx,
    grad,
    near,
    split,
)

# ── UFL helper functions ────────────────────────────────────────────────────

def _h(phi):
    """Interpolation function h(φ) = -2φ³ + 3φ²."""
    return -2.0 * phi**3 + 3.0 * phi**2


def _hp(phi):
    """Derivative h'(φ) = -6φ² + 6φ."""
    return -6.0 * phi**2 + 6.0 * phi


def _gp(phi):
    """Derivative of double-well g(φ) = φ²(1-φ)²: g'(φ) = 4φ³ - 6φ² + 2φ."""
    return 4.0 * phi**3 - 6.0 * phi**2 + 2.0 * phi


# ── Initial conditions ──────────────────────────────────────────────────────

class _InitialConditions(UserExpression):
    """Mixed initial condition (φ₀, c₀) with tanh profile for φ."""

    def __init__(self, k, c_se, c_le, center=0.5, **kwargs):
        super().__init__(**kwargs)
        self.k = k
        self.c_se = c_se
        self.c_le = c_le
        self.center = center

    def eval(self, value, x):
        phi = 0.5 * (1.0 - np.tanh(self.k * (x[0] - self.center)))
        hp = -2.0 * phi**3 + 3.0 * phi**2
        c = hp * self.c_se
        value[0] = phi
        value[1] = c

    def value_shape(self):
        return (2,)


# ── Boundary markers ────────────────────────────────────────────────────────

def _left(x, on_boundary):
    return on_boundary and near(x[0], 0.0)


def _right(x, on_boundary):
    return on_boundary and near(x[0], 1.0)


# ── Non-dimensional parameters ──────────────────────────────────────────────

def compute_nondim_params(physical, L_c, T_c):
    """Return {N_CH, N_AC1, N_AC2, N_AC3} from physical params and scales."""
    p = physical
    return {
        'N_CH':  p.D * T_c / L_c**2,
        'N_AC1': 2.0 * p.A * p.L * T_c,
        'N_AC2': p.L * p.omega_phi * T_c,
        'N_AC3': p.L * p.alpha_phi * T_c / L_c**2,
    }


# ── MPI-aware solution gathering ───────────────────────────────────────────

def _gather_solution(sol_func, comm):
    """Gather φ and c DOF values + coords to rank 0, deduplicating ghost DOFs."""
    phi_f, c_f = sol_func.split(deepcopy=True)
    V_phi = phi_f.function_space()
    local_coords = V_phi.tabulate_dof_coordinates().ravel()
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
        all_coords = np.concatenate(coords_list)
        all_phi = np.concatenate(phi_list)
        all_c = np.concatenate(c_list)
        unique_idx = np.unique(all_coords, return_index=True)[1]
        return all_coords[unique_idx], all_phi[unique_idx], all_c[unique_idx]
    return None, None, None


# ── Output ──────────────────────────────────────────────────────────────────

def _save_results(output_dir, filename, coords, times, phi_hist, c_hist):
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    np.savez_compressed(filepath, mesh=coords, time=times, phi=phi_hist, c=c_hist)
    print(f"Saved {filepath}  ({phi_hist.shape[0]} steps, {phi_hist.shape[1]} dofs)")


def _plot_snapshots(output_dir, coords, times, phi_hist, c_hist, n_snapshots=6):
    import matplotlib.pyplot as plt

    n_total = len(times)
    indices = np.linspace(0, n_total - 1, n_snapshots, dtype=int)

    fig, ax = plt.subplots(figsize=(8, 4))

    colors = plt.cm.viridis(np.linspace(0, 1, n_snapshots))
    for i, idx in enumerate(indices):
        t = times[idx]
        ax.plot(coords, phi_hist[idx], color=colors[i], linestyle='-',
                label=f'φ  t*={t:.2f}')
        ax.plot(coords, c_hist[idx], color=colors[i], linestyle='--',
                label=f'c  t*={t:.2f}')

    ax.set_xlabel(r'$x^*$')
    ax.set_ylabel(r'$\phi,\ c$')
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=7)
    fig.tight_layout()

    filepath = os.path.join(output_dir, 'snapshots.png')
    fig.savefig(filepath, dpi=150)
    plt.close(fig)
    print(f"Saved {filepath}")


def _plot_heatmaps(output_dir, coords, times, phi_hist, c_hist):
    import matplotlib.pyplot as plt

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 4))

    im0 = ax0.pcolormesh(coords, times, phi_hist, cmap='coolwarm',
                          vmin=0, vmax=1, shading='nearest')
    ax0.set_xlabel(r'$x^*$')
    ax0.set_ylabel(r'$t^*$')
    ax0.set_title(r'$\phi$')
    plt.colorbar(im0, ax=ax0)

    im1 = ax1.pcolormesh(coords, times, c_hist, cmap='coolwarm',
                          vmin=0, vmax=1, shading='nearest')
    ax1.set_xlabel(r'$x^*$')
    ax1.set_ylabel(r'$t^*$')
    ax1.set_title(r'$c$')
    plt.colorbar(im1, ax=ax1)

    fig.tight_layout()
    filepath = os.path.join(output_dir, 'heatmaps.png')
    fig.savefig(filepath, dpi=150)
    plt.close(fig)
    print(f"Saved {filepath}")


# ── Main solver ─────────────────────────────────────────────────────────────

def solve(config):
    comm = _MPI.COMM_WORLD
    rank = comm.rank

    # ── non-dimensional parameters ──────────────────────────────────────
    phys = config.physical
    L_c = config.nondim.L_c
    T_c = config.nondim.T_c

    params = compute_nondim_params(phys, L_c, T_c)
    N_CH  = params['N_CH']
    N_AC1 = params['N_AC1']
    N_AC2 = params['N_AC2']
    N_AC3 = params['N_AC3']
    dc_se_le = phys.c_se - phys.c_le

    if rank == 0:
        print(f"N_CH={N_CH:.4e}  N_AC1={N_AC1:.4e}  N_AC2={N_AC2:.4e}  N_AC3={N_AC3:.4e}")

    # ── mesh & function space ───────────────────────────────────────────
    mesh = IntervalMesh(comm, config.mesh.n_cells, 0.0, 1.0)
    P1 = FiniteElement('CG', mesh.ufl_cell(), 1)
    V = FunctionSpace(mesh, P1 * P1)

    sol = Function(V)
    sol_prev = Function(V)

    # ── initial condition ───────────────────────────────────────────────
    k = np.sqrt(phys.omega_phi / (2.0 * phys.alpha_phi)) * L_c
    ic = _InitialConditions(k=k, c_se=phys.c_se, c_le=phys.c_le,
                             center=config.init.center, degree=1)
    sol.interpolate(ic)
    sol_prev.interpolate(ic)

    # ── split fields ────────────────────────────────────────────────────
    phi, c = split(sol)
    phi_prev, c_prev = split(sol_prev)
    v_phi, v_c = TestFunctions(V)
    dt_const = Constant(config.time.dt)

    # ── boundary conditions ─────────────────────────────────────────────
    bcs = [
        DirichletBC(V.sub(0), Constant(1.0), _left),
        DirichletBC(V.sub(0), Constant(0.0), _right),
        DirichletBC(V.sub(1), Constant(1.0), _left),
        DirichletBC(V.sub(1), Constant(0.0), _right),
    ]

    # ── time integration method ───────────────────────────────────────────
    # theta = 1.0: backward Euler, theta = 0.5: Crank-Nicolson
    theta = config.time.theta

    # ── weak form (non-dimensional) ─────────────────────────────────────
    c_th = theta * c + (1.0 - theta) * c_prev
    phi_th = theta * phi + (1.0 - theta) * phi_prev

    # Cahn-Hilliard-type equation for c
    F_c = (c - c_prev) / dt_const * v_c * dx \
        + N_CH * dot(grad(c_th), grad(v_c)) * dx \
        - N_CH * dc_se_le * dot(grad(_h(phi_th)), grad(v_c)) * dx

    # Allen-Cahn equation for φ
    F_phi = (phi - phi_prev) / dt_const * v_phi * dx \
        + N_AC3 * dot(grad(phi_th), grad(v_phi)) * dx \
        + N_AC2 * _gp(phi_th) * v_phi * dx \
        - N_AC1 * (c_th - _h(phi_th) * dc_se_le - phys.c_le) * dc_se_le * _hp(phi_th) * v_phi * dx

    F = F_c + F_phi
    J = derivative(F, sol, TrialFunction(V))

    # ── nonlinear solver ────────────────────────────────────────────────
    problem = NonlinearVariationalProblem(F, sol, bcs, J)
    solver = NonlinearVariationalSolver(problem)

    prm = solver.parameters['newton_solver']
    prm['absolute_tolerance'] = config.solver.atol
    prm['relative_tolerance'] = config.solver.rtol
    prm['maximum_iterations'] = config.solver.max_iters
    prm['linear_solver'] = 'mumps'
    prm['convergence_criterion'] = 'incremental'

    # ── time stepping ───────────────────────────────────────────────────
    dt = config.time.dt
    n_steps = int(config.time.t_final / dt)
    times = np.linspace(dt, config.time.t_final, n_steps)

    if rank == 0:
        print(f"Domain: x* ∈ [0,1],  cells={config.mesh.n_cells},  "
              f"dt*={dt},  steps={n_steps}")

    # Record initial condition
    coords, phi_vals, c_vals = _gather_solution(sol, comm)
    if rank == 0:
        phi_history = [phi_vals]
        c_history = [c_vals]
        time_history = [0.0]

    # ── time loop ───────────────────────────────────────────────────────
    for step, t in enumerate(times, 1):
        sol_prev.assign(sol)
        solver.solve()

        coords, phi_vals, c_vals = _gather_solution(sol, comm)
        if rank == 0:
            phi_history.append(phi_vals)
            c_history.append(c_vals)
            time_history.append(t)
            print(f"  Step {step}/{n_steps}  t*={t:.4f}")

    # ── save ────────────────────────────────────────────────────────────
    if rank == 0:
        _save_results(config.output.dir, config.output.filename,
                      coords, np.array(time_history),
                      np.array(phi_history), np.array(c_history))
        _plot_snapshots(config.output.dir, coords, np.array(time_history),
                        np.array(phi_history), np.array(c_history))
        _plot_heatmaps(config.output.dir, coords, np.array(time_history),
                       np.array(phi_history), np.array(c_history))

    # Force FEniCS object cleanup before MPI is finalized (avoids harmless
    # but noisy "MPI_Comm_get_attr after MPI_FINALIZE" errors).
    del solver, problem, sol, sol_prev, mesh, V, bcs, F, J
    gc.collect()
    comm.Barrier()


# ── Config ───────────────────────────────────────────────────────────────────

_DEFAULT_CONFIG = os.path.join(os.path.dirname(__file__), 'configs', 'solver_activation.py')
_ = _config_flags.DEFINE_config_file('config', _DEFAULT_CONFIG, 'Config file')
FLAGS = _absl_app.flags.FLAGS


# ── Entry point ─────────────────────────────────────────────────────────────

def _main(_):
    solve(FLAGS.config)


if __name__ == '__main__':
    _absl_app.run(_main)
