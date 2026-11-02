"""Independent manufactured Neumann/traction checks on the actual tagged mesh."""
import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ['PKG_CONFIG_PATH'] = str(Path(sys.prefix) / 'lib/pkgconfig')
import dolfin as df
import numpy as np
from ml_collections import ConfigDict
from examples.channel_nsch2d.solver import mesh_and_markers


def check(config, output):
    c = ConfigDict(json.loads(Path(config).read_text()))
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    df.set_log_level(40)
    rows = []
    for nx, ny in ((16, 8), (32, 16)):
        mesh, tags = mesh_and_markers(c, (c.geometry.upper, c.geometry.lower), nx, ny)
        normal = df.FacetNormal(mesh)
        ds = df.Measure('ds', domain=mesh, subdomain_data=tags)
        dx = df.Measure('dx', domain=mesh, metadata={'quadrature_degree': 6})
        length = [float(df.assemble(df.Constant(1.) * ds(i))) for i in (1, 2, 3)]
        if min(length) <= 0 or abs(length[1] - c.geometry.height) > 1e-10 or abs(length[2] - c.geometry.height) > 1e-10:
            raise ValueError('Wall/inlet/outlet marker measures are inconsistent')
        closure = [float(df.assemble(normal[i] * ds)) for i in range(2)]
        V = df.FunctionSpace(mesh, 'CG', 2)
        trial, test = df.TrialFunction(V), df.TestFunction(V)
        X = df.SpatialCoordinate(mesh)
        exact = X[0]**2 + X[1]**2
        # -Delta(exact)=-4, g=grad(exact).n on the two natural-boundary groups.
        g = df.dot(df.as_vector((2*X[0], 2*X[1])), normal)
        phi = df.Function(V)
        inlet = df.Expression('x[0]*x[0]+x[1]*x[1]', degree=2)
        df.solve(df.inner(df.grad(trial), df.grad(test))*dx ==
                 -4*test*dx + g*test*(ds(1) + ds(3)), phi,
                 df.DirichletBC(V, inlet, tags, 2),
                 solver_parameters={'linear_solver': c.reference.linear_solver})
        phi_error = float(np.sqrt(df.assemble((phi - exact)**2 * dx)))
        flux_error = float(np.sqrt(df.assemble((df.dot(df.grad(phi), normal) - g)**2 * (ds(1) + ds(3)))))
        cell = mesh.ufl_cell()
        W = df.FunctionSpace(mesh, df.MixedElement((df.VectorElement('CG', cell, 2),
                                                    df.FiniteElement('CG', cell, 1))))
        (u, p), (v, q) = df.TrialFunctions(W), df.TestFunctions(W)
        nu = c.physical.viscosity
        a = (2*nu*df.inner(df.sym(df.grad(u)), df.sym(df.grad(v))) - p*df.div(v) + q*df.div(u))*dx
        # u=(y,0), p=2*nu*x. Then -div(2nu eps(u)-pI)=(2nu,0).
        # At the vertical outlet traction=(-2nu*L,nu), fixing pressure's constant.
        force = df.Constant((2*nu, 0.))
        traction = df.Constant((-2*nu*c.geometry.length, nu))
        rhs = df.inner(force, v)*dx + df.inner(traction, v)*ds(3)
        prescribed = df.Expression(('x[1]', '0.'), degree=1)
        bcs = [df.DirichletBC(W.sub(0), prescribed, tags, i) for i in (1, 2)]
        solution = df.Function(W)
        df.solve(a == rhs, solution, bcs, solver_parameters={'linear_solver': c.reference.linear_solver})
        velocity, pressure = solution.split(deepcopy=True)
        velocity_error = float(np.sqrt(df.assemble(df.inner(velocity - df.as_vector((X[1], 0.)),
                                                            velocity - df.as_vector((X[1], 0.)))*dx)))
        pressure_error = float(np.sqrt(df.assemble((pressure - 2*nu*X[0])**2 * dx)))
        stress = 2*nu*df.sym(df.grad(velocity)) - pressure*df.Identity(2)
        traction_error = float(np.sqrt(df.assemble(df.inner(df.dot(stress, normal) - traction,
                                                            df.dot(stress, normal) - traction)*ds(3))))
        row = dict(grid=[nx, ny], boundary_lengths=length, normal_integral_closure=closure,
                   poisson_L2=phi_error, poisson_neumann_L2=flux_error, velocity_L2=velocity_error,
                   pressure_L2=pressure_error, outlet_traction_L2=traction_error)
        if max(phi_error, flux_error, velocity_error, pressure_error, traction_error, *np.abs(closure)) > 1e-8:
            raise ValueError(f'Manufactured boundary test failed: {row}')
        rows.append(row)
    report = dict(status='manufactured_Neumann_traction_and_pressure_constant_checks_passed', rows=rows,
                  scope='Actual curved mesh/marker factory; nonzero manufactured boundary data, no transient NSCH solve; checks implementation conventions, not complete physical-model correctness')
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    check(args.config, args.output)
