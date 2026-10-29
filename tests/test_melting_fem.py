"""Independent fixed-physics, weak-form and MPI evaluation checks (fenicsenv)."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

HAS_FENICS = importlib.util.find_spec('dolfin') is not None
if HAS_FENICS:
    import dolfin as df
    from mpi4py import MPI
    from examples.icemelting3d.configs.solver_base import get_config
    from examples.icemelting3d.configs.train_operator import get_config as operator_config
    from examples.icemelting3d.solver import initial_field, load_problem, make_imex_step, query_plan, sample_field, weak_residual


@unittest.skipUnless(HAS_FENICS, 'Local FEM tests require the separate fenicsenv environment')
class MeltingFEMTests(unittest.TestCase):
    def test_fixed_physics_and_exact_initial_geometry(self):
        c = operator_config()
        parameters = [.5, .5, .5, .3, .3, .3, 1., 0., 0., 0.]
        with tempfile.TemporaryDirectory() as folder:
            config_path = Path(folder)/'training.json'
            manifest_path = Path(folder)/'manifest.json'
            config_path.write_text(c.to_json())
            manifest_path.write_text(json.dumps(dict(test_indices=[5], parameters=[parameters]*6,
                identity={key: c[key].to_dict() for key in ('physical', 'nondim', 'initial')})))
            p = load_problem(config_path, manifest_path, 5)
            with self.assertRaises(ValueError):
                load_problem(config_path, manifest_path, 0)
        self.assertEqual(p['physical']['eps_star'], .0225)
        n1 = .1 * 5 / 100**2
        np.testing.assert_allclose(p['coefficients'], [n1, n1/.0225**2, .25/(np.sqrt(2)*.0225)], rtol=1e-15)
        self.assertNotIn('jax', sys.modules)
        points = np.array([[.5, .5, .5], [.8, .5, .5], [.2, .1, .3], [1., 1., 1.]])
        expected = np.tanh((.3-np.linalg.norm(points-.5, axis=1))/(np.sqrt(2)*.0225))
        np.testing.assert_allclose(initial_field(points, p), expected, rtol=0, atol=2e-14)

    def test_weak_diffusion_sign_and_physical_reaction(self):
        mesh = df.UnitCubeMesh(MPI.COMM_WORLD, 2, 2, 2)
        measure = df.dx(domain=mesh, metadata={'quadrature_degree': 8})
        # This polynomial has zero normal derivative on every face.
        x = df.SpatialCoordinate(mesh)
        phi = x[0]**2*(1-x[0])**2
        v = x[0]**2
        result = df.assemble(weak_residual(phi, phi, v, .01, [1, 0, 0], measure))
        self.assertAlmostEqual(result, -1/15, places=12)
        p, old, dt = .2, .1, .005
        coeff = [.00005, .0987654321, 7.856742013]
        mid = (p+old)/2
        expected = (p-old)/dt+coeff[1]*(mid**3-mid)+coeff[2]*(1-mid**2)
        V = df.FunctionSpace(mesh, 'CG', 2)
        fields = [df.interpolate(df.Constant(value), V) for value in (p, old, 1)]
        form = weak_residual(*fields, dt, coeff, measure)
        self.assertAlmostEqual(df.assemble(form), expected, places=10)

    def test_mpi_query_values_and_owned_dof_initialization(self):
        mesh = df.UnitCubeMesh(MPI.COMM_WORLD, 3, 3, 3)
        V = df.FunctionSpace(mesh, 'CG', 2)
        field = df.Function(V)
        x = V.tabulate_dof_coordinates().reshape(-1, 3)[:field.vector().local_size()]
        field.vector().set_local(x[:, 0]**2+2*x[:, 1]+x[:, 2])
        field.vector().apply('insert')
        points = np.array([[0, 0, 0], [1, 1, 1], [.3, .7, .2], [.8, .1, .9], [.5, .5, .5]])
        actual = sample_field(field, points, query_plan(mesh, points), MPI.COMM_WORLD)
        if MPI.COMM_WORLD.rank == 0:
            np.testing.assert_allclose(actual, points[:, 0]**2+2*points[:, 1]+points[:, 2], atol=2e-14)

    def test_imex_temporal_order_against_independent_homogeneous_ode(self):
        from scipy.integrate import solve_ivp
        c = get_config()
        coefficients = [.00005, .0987654321, 7.856742013]
        exact = solve_ivp(lambda t, y: -coefficients[1]*(y**3-y)-coefficients[2]*(1-y**2),
                         (0, .1), [.2], method='DOP853', rtol=1e-12, atol=1e-14).y[0, -1]
        mesh = df.UnitCubeMesh(MPI.COMM_WORLD, 2, 2, 2)
        V = df.FunctionSpace(mesh, 'CG', 2)
        measure = df.dx(domain=mesh, metadata={'quadrature_degree': 8})
        errors = []
        for dt in (.01, .005, .0025):
            phi = df.interpolate(df.Constant(.2), V)
            advance = make_imex_step(phi, V, dt, coefficients, measure, c.solver)
            for step in range(1, round(.1/dt)+1):
                advance(step)
            error = float(df.assemble((phi-exact)**2*measure))**.5
            errors.append(error)
        self.assertTrue(3.5 < errors[0]/errors[1] < 4.5, errors)
        self.assertTrue(3.5 < errors[1]/errors[2] < 4.5, errors)

    def test_imex_diffusion_equations_mass_and_energy(self):
        c = get_config()
        mesh = df.UnitCubeMesh(MPI.COMM_WORLD, 3, 3, 3)
        V = df.FunctionSpace(mesh, 'CG', 2)
        phi = df.interpolate(df.Expression('x[0]*x[0]+2*x[1]+x[2]', degree=2), V)
        initial = phi.copy(deepcopy=True)
        measure = df.dx(domain=mesh, metadata={'quadrature_degree': 8})
        mass = float(df.assemble(phi*measure))
        energy = float(df.assemble(phi**2*measure))
        dt, n1 = .01, .02
        # Symbolic zero coefficients retain the vector form's UFL rank;
        # literal zeros simplify the entire reaction form to a scalar zero.
        advance = make_imex_step(phi, V, dt, [n1, df.Constant(0), df.Constant(0)], measure, c.solver)
        test = df.TestFunction(V)
        advance(1)
        form = ((phi-initial)*test/dt+n1*df.dot(df.grad(phi), df.grad(test)))*measure
        self.assertLess(df.assemble(form).norm('l2'), 1e-8)
        previous = phi.copy(deepcopy=True)
        advance(2)
        form = ((1.5*phi-2*previous+.5*initial)*test/dt
                +n1*df.dot(df.grad(phi), df.grad(test)))*measure
        self.assertLess(df.assemble(form).norm('l2'), 1e-8)
        self.assertAlmostEqual(float(df.assemble(phi*measure)), mass, places=10)
        self.assertLess(float(df.assemble(phi**2*measure)), energy)


if __name__ == '__main__':
    unittest.main()
