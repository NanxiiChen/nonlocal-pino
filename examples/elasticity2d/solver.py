"""2D FEM solver for linear elastic plate with circular hole (plane strain).

Left edge clamped (u_x=u_y=0), right edge displacement-controlled (u_x=u_load).
Steady-state linear elasticity.

Environment:
  conda activate fenicsenv

Usage:
  python examples/elasticity2d/solver.py
  python examples/elasticity2d/solver.py --config=examples/elasticity2d/configs/solver_base.py
"""

import os, subprocess, sys

# ── Monkey-patch UFL for Python 3.14 compat ────────────────────────────────
# The _afmt function in check_arities.py expects an iterable of (arg, conj)
# pairs but is sometimes called with a single pair.  Normalise input.
import ufl.algorithms.check_arities as _ufl_ca
_orig_afmt = _ufl_ca._afmt
def _patched_afmt(atuple):
    if len(atuple) > 0 and not isinstance(atuple[0], tuple):
        atuple = (atuple,)
    return _orig_afmt(atuple) if atuple else ()
_ufl_ca._afmt = _patched_afmt
# ───────────────────────────────────────────────────────────────────────────

_pkgconfig_dir = '/root/miniconda3/envs/fenicsenv/lib/pkgconfig'
if not os.path.isdir(_pkgconfig_dir):
    _pkgconfig_dir = os.path.join(sys.prefix, 'lib', 'pkgconfig')
if os.path.isdir(_pkgconfig_dir):
    os.environ.setdefault('PKG_CONFIG_PATH', '')
    if _pkgconfig_dir not in os.environ['PKG_CONFIG_PATH']:
        os.environ['PKG_CONFIG_PATH'] = _pkgconfig_dir + ':' + os.environ['PKG_CONFIG_PATH']

import numpy as np
from mpi4py import MPI as _MPI

from fenics import (
    Constant, DirichletBC, FiniteElement, Function, FunctionSpace,
    Mesh, SubDomain, TestFunction, TrialFunction,
    XDMFFile, assemble, dot, dx, inner, near, sym, grad, project,
    VectorFunctionSpace, LUSolver,
)
from ufl import Identity

from absl import app as _absl_app
from ml_collections import config_flags as _config_flags

_DEFAULT = os.path.join(os.path.dirname(__file__), 'configs', 'solver_base.py')
_CONFIG = _config_flags.DEFINE_config_file('config', _DEFAULT, 'Config file')
FLAGS = _absl_app.flags.FLAGS


# ── Mesh ─────────────────────────────────────────────────────────────────────

def _generate_mesh(geo_file, msh_file):
    """Run Gmsh + meshio conversion to XDMF."""
    import meshio
    _geo_abs = os.path.abspath(geo_file)
    _msh_abs = os.path.abspath(msh_file)
    subprocess.run(
        ["gmsh", "-2", _geo_abs, "-o", _msh_abs],
        check=True, capture_output=True)
    msh = meshio.read(_msh_abs)
    tri_cells = None
    for cb in msh.cells:
        if cb.type == "triangle":
            tri_cells = cb.data
            break
    _xdmf = _msh_abs.replace('.msh', '.xdmf')
    meshio.write(_xdmf, meshio.Mesh(
        points=msh.points[:, :2],
        cells=[("triangle", tri_cells)],
    ))
    return _xdmf


# ── Solver ───────────────────────────────────────────────────────────────────

def solve(config):
    comm = _MPI.COMM_WORLD
    rank = comm.rank

    Lx = config.geometry.Lx
    Ly = config.geometry.Ly
    lam = config.physical.lam
    mu = config.physical.mu
    u_load = config.load.u_load

    if rank == 0:
        print(f"Lx={Lx}, Ly={Ly}, lam*={lam}, mu*={mu}, u_load={u_load}")

    # ── Mesh ──────────────────────────────────────────────────────────────
    if rank == 0:
        xdmf_file = _generate_mesh(config.mesh.geo_file, config.mesh.msh_file)
    comm.Barrier()
    xdmf_file = os.path.abspath(config.mesh.msh_file).replace('.msh', '.xdmf')
    mesh = Mesh()
    with XDMFFile(xdmf_file) as xf:
        xf.read(mesh)

    if rank == 0:
        print(f"Mesh: {mesh.num_vertices()} vertices, {mesh.num_cells()} cells")

    # ── Function space ────────────────────────────────────────────────────
    V = VectorFunctionSpace(mesh, 'CG', 1)
    u = TrialFunction(V)
    v = TestFunction(V)
    sol = Function(V)

    # ── Weak form: plane strain linear elasticity ─────────────────────────
    # ∫ [2μ ε(u):ε(v) + λ (∇·u)(∇·v)] dx = 0
    from ufl import div
    eps_u = sym(grad(u))
    eps_v = sym(grad(v))
    a = (2.0 * mu * inner(eps_u, eps_v) + lam * div(u) * div(v)) * dx
    L_form = inner(Constant((0.0, 0.0)), v) * dx  # zero body force

    # ── BC subdomains (defined here to capture Lx) ───────────────────────
    class _LeftEdge(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and near(x[0], 0.0)

    class _RightEdge(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and near(x[0], Lx)

    # ── BCs ───────────────────────────────────────────────────────────────
    left = _LeftEdge()
    right = _RightEdge()

    bc_left_x = DirichletBC(V.sub(0), Constant(0.0), left)    # u_x = 0 on left
    bc_left_y = DirichletBC(V.sub(1), Constant(0.0), left)    # u_y = 0 on left
    bc_right_x = DirichletBC(V.sub(0), Constant(u_load), right)  # u_x = u_load on right
    bc_right_y = DirichletBC(V.sub(1), Constant(0.0), right)  # u_y = 0 on right

    bcs = [bc_left_x, bc_left_y, bc_right_x, bc_right_y]

    # ── Solve ─────────────────────────────────────────────────────────────
    A = assemble(a)
    b = assemble(L_form)
    for bc in bcs:
        bc.apply(A, b)
    _solver = LUSolver(A, 'default')
    _solver.solve(sol.vector(), b)

    if rank == 0:
        print(f"Solve complete. u_x max={sol.vector().max():.6f}, "
              f"min={sol.vector().min():.6f}")

    # ── Extract and save ──────────────────────────────────────────────────
    u_x_f, u_y_f = sol.split(deepcopy=True)

    # Project stress
    P1 = FiniteElement('CG', mesh.ufl_cell(), 1)
    S = FunctionSpace(mesh, P1)
    from ufl import div as _div_op
    def _sigma(u_):
        return 2.0 * mu * sym(grad(u_)) + lam * _div_op(u_) * Identity(2)
    s_xx = project(_sigma(sol)[0, 0], S)
    s_yy = project(_sigma(sol)[1, 1], S)
    s_xy = project(_sigma(sol)[0, 1], S)

    if rank == 0:
        coords = S.tabulate_dof_coordinates().reshape(-1, 2)
        ux_vals = u_x_f.vector().get_local()
        uy_vals = u_y_f.vector().get_local()
        sx_vals = s_xx.vector().get_local()
        sy_vals = s_yy.vector().get_local()
        sxy_vals = s_xy.vector().get_local()

        out_dir = config.output.dir
        os.makedirs(out_dir, exist_ok=True)
        fpath = os.path.join(out_dir, config.output.filename)
        np.savez_compressed(fpath,
                            coords=coords,
                            u_x=ux_vals, u_y=uy_vals,
                            sigma_xx=sx_vals, sigma_yy=sy_vals,
                            sigma_xy=sxy_vals)
        print(f"Saved {fpath}")
        print(f"  coords: {coords.shape}, u_x range: [{ux_vals.min():.6f}, {ux_vals.max():.6f}]")


# ── Entry ────────────────────────────────────────────────────────────────────

def _main(_):
    solve(FLAGS.config)


if __name__ == '__main__':
    _absl_app.run(_main)
