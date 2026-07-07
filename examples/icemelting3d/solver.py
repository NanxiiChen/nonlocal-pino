"""3D FEM solver for ice melting — non-dimensional form with Crank-Nicolson.

∂φ/∂t* = N₁ Δ*φ - N₂ (φ³ - φ) - N₃ (1 - φ²)

Domain: Ω* = [0,1]³,  t* ∈ [0,1]
Analytical: φ(x*,t*) = tanh((R₀* - λ* t* - |x* - xc*|) / (√2 ε*))

Usage:
  conda activate fenicsenv
  mpirun --allow-run-as-root -np 12 python examples/icemelting3d/solver.py
"""

import gc, os, sys
import numpy as np

_fenics_prefix = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
_pkgconfig_dir = '/root/miniconda3/envs/fenicsenv/lib/pkgconfig'
if not os.path.isdir(_pkgconfig_dir):
    _pkgconfig_dir = os.path.join(sys.prefix, 'lib', 'pkgconfig')
if os.path.isdir(_pkgconfig_dir):
    os.environ.setdefault('PKG_CONFIG_PATH', '')
    if _pkgconfig_dir not in os.environ['PKG_CONFIG_PATH']:
        os.environ['PKG_CONFIG_PATH'] = _pkgconfig_dir + ':' + os.environ['PKG_CONFIG_PATH']

from absl import app as _absl_app
from ml_collections import config_flags as _config_flags
from fenics import (
    BoxMesh, Constant, FiniteElement, Function, FunctionSpace,
    NonlinearVariationalProblem, NonlinearVariationalSolver, Point,
    TestFunction, TrialFunction, UserExpression,
    derivative, dot, dx, grad,
)


# ── Initial condition ──────────────────────────────────────────────────────

class _IceSphereIC(UserExpression):
    """φ = tanh((R₀* - |x* - xc*|) / (√2 ε*))"""

    def __init__(self, R0_star, xc, eps_star, **kwargs):
        super().__init__(**kwargs)
        self.R0_star = R0_star
        self.xc = xc
        self.eps_star = eps_star
        self._denom = np.sqrt(2.0) * eps_star

    def eval(self, value, x):
        R = np.sqrt((x[0] - self.xc[0])**2
                  + (x[1] - self.xc[1])**2
                  + (x[2] - self.xc[2])**2)
        value[0] = np.tanh((self.R0_star - R) / self._denom)

    def value_shape(self):
        return ()


# ── Solver ─────────────────────────────────────────────────────────────────

def solve(config):
    comm = _MPI.COMM_WORLD
    rank = comm.rank

    phys = config.physical
    M      = phys.M
    lam    = phys.lam
    R0     = phys.R0
    L_c    = config.nondim.L_c
    T_c    = config.nondim.T_c
    N      = config.mesh.N
    h_dim  = 100.0 / N               # dimensional grid spacing
    eps_dim = 6.0 * h_dim / (2.0 * np.sqrt(2.0) * np.arctanh(0.9))
    h_star = 1.0 / N                 # non-dimensional grid spacing
    eps_star = eps_dim / L_c

    # Non-dimensional parameters
    R0_star = R0 / L_c               # 0.35
    lam_star = lam * T_c / L_c       # 0.15
    N1 = M * T_c / L_c**2
    N2 = N1 / eps_star**2
    N3 = lam_star / (np.sqrt(2.0) * eps_star)
    dt_star = config.time.dt
    T_star  = config.time.T
    denom = np.sqrt(2.0) * eps_star

    if rank == 0:
        print(f"M={M}  lam={lam}  R0={R0}  N={N}")
        print(f"L_c={L_c}  T_c={T_c}  h*={h_star:.4f}  eps*={eps_star:.4f}")
        print(f"R0*={R0_star:.4f}  lam*={lam_star:.4f}")
        print(f"N1={N1:.6e}  N2={N2:.6e}  N3={N3:.6e}")
        print(f"dt*={dt_star}  T*={T_star}  theta={config.time.theta}")

    # ── mesh (unit cube) ───────────────────────────────────────────────────
    mesh = BoxMesh(comm,
                   Point(0.0, 0.0, 0.0),
                   Point(1.0, 1.0, 1.0),
                   N, N, N)
    P1 = FiniteElement('CG', mesh.ufl_cell(), 1)
    V = FunctionSpace(mesh, P1)

    phi = Function(V)
    phi_prev = Function(V)

    # ── IC (sphere centred at (0.5, 0.5, 0.5)) ─────────────────────────────
    xc = (0.5, 0.5, 0.5)
    ic_expr = _IceSphereIC(R0_star=R0_star, xc=xc, eps_star=eps_star, degree=2)
    phi.interpolate(ic_expr)
    phi_prev.interpolate(ic_expr)

    v = TestFunction(V)
    dt_const = Constant(dt_star)
    theta = config.time.theta

    phi_th = theta * phi + (1.0 - theta) * phi_prev

    # Weak form (non-dimensional):
    # ∫ ∂φ/∂t* v + N₁∫∇*φ_th·∇*v + N₂∫(φ_th³-φ_th)v + N₃∫(1-φ_th²)v = 0
    F = (phi - phi_prev) / dt_const * v * dx \
        + N1 * dot(grad(phi_th), grad(v)) * dx \
        + N2 * (phi_th**3 - phi_th) * v * dx \
        + N3 * (1.0 - phi_th * phi_th) * v * dx

    J = derivative(F, phi, TrialFunction(V))

    # ── solver ────────────────────────────────────────────────────────────
    problem = NonlinearVariationalProblem(F, phi, [], J)
    solver = NonlinearVariationalSolver(problem)
    prm = solver.parameters['newton_solver']
    prm['absolute_tolerance'] = config.solver.atol
    prm['relative_tolerance'] = config.solver.rtol
    prm['maximum_iterations'] = config.solver.max_iters
    prm['convergence_criterion'] = 'incremental'
    prm['linear_solver'] = 'mumps'
    prm['relaxation_parameter'] = 0.9

    # ── initial save ──────────────────────────────────────────────────────
    save_every = config.output.save_every
    phi_vals = phi.vector().get_local()
    coords = V.tabulate_dof_coordinates().reshape(-1, 3)

    # Analytical solution at t*=0
    R_analytic = R0_star
    radii = np.sqrt(np.sum((coords - np.array(xc))**2, axis=1))
    phi_analytic = np.tanh((R_analytic - radii) / denom)
    err = np.sqrt(np.mean((phi_vals - phi_analytic)**2))
    denom_err = np.sqrt(np.mean(phi_analytic**2))
    rel_err = err / denom_err if denom_err > 1e-12 else 0.0

    if rank == 0:
        print(f"  t*=0.0000  R*={R_analytic:.4f}  rel_L2={rel_err:.6e}")
        t_hist = [0.0]
        phi_hist = [phi_vals]
        err_hist = [rel_err]

    # ── time stepping ─────────────────────────────────────────────────────
    t_current = 0.0
    next_save = save_every
    step = 0

    while t_current < T_star - 1e-12:
        step += 1
        t_current += dt_star
        t_current = min(t_current, T_star)

        phi_prev.assign(phi)
        solver.solve()

        # Compute error every step (numpy, negligible cost vs FEM solve)
        phi_vals = phi.vector().get_local()
        R_analytic = R0_star - lam_star * t_current
        phi_analytic = np.tanh((R_analytic - radii) / denom)
        err = np.sqrt(np.mean((phi_vals - phi_analytic)**2))
        denom_err = np.sqrt(np.mean(phi_analytic**2))
        rel_err = err / denom_err if denom_err > 1e-12 else 0.0

        if rank == 0:
            print(f"  t*={t_current:.4f}  R*={R_analytic:.4f}  "
                  f"rel_L2={rel_err:.6e}")

        if t_current >= next_save - 1e-12 or t_current >= T_star - 1e-12:
            if rank == 0:
                t_hist.append(t_current)
                phi_hist.append(phi_vals)
                err_hist.append(rel_err)
            next_save += save_every

    # ── save ──────────────────────────────────────────────────────────────
    if rank == 0:
        os.makedirs(config.output.dir, exist_ok=True)
        t_arr = np.array(t_hist)
        phi_arr = np.array(phi_hist)
        err_arr = np.array(err_hist)

        fpath = os.path.join(config.output.dir, config.output.filename)
        np.savez_compressed(fpath,
                            coords=coords, time=t_arr,
                            phi=phi_arr, rel_error=err_arr,
                            params={'M': M, 'lam': lam, 'R0': R0,
                                    'L_c': L_c, 'T_c': T_c,
                                    'eps_star': eps_star, 'N': N,
                                    'N1': N1, 'N2': N2, 'N3': N3,
                                    'dt_star': dt_star,
                                    'theta': config.time.theta})
        print(f"Saved {fpath}")
        print(f"  snapshots={len(t_arr)}  final rel_L2={err_arr[-1]:.6e}")

    del solver, problem, phi, phi_prev, mesh, V, F, J
    gc.collect()
    comm.Barrier()


# ── MPI safeguard ───────────────────────────────────────────────────────────

try:
    from mpi4py import MPI as _MPI
except ImportError:
    _MPI = type('_FakeMPI', (), {
        'COMM_WORLD': type('_FakeComm', (), {
            'rank': 0, 'size': 1, 'Barrier': lambda self: None,
            'gather': lambda self, data, root=0: [data],
        })(),
    })()


# ── Config ─────────────────────────────────────────────────────────────────

_DEFAULT = os.path.join(os.path.dirname(__file__), 'configs', 'solver_base.py')
_CONFIG = _config_flags.DEFINE_config_file('config', _DEFAULT, 'Config file')
FLAGS = _absl_app.flags.FLAGS


def _main(_):
    solve(FLAGS.config)


if __name__ == '__main__':
    _absl_app.run(_main)
