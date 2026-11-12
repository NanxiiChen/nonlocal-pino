"""Independent radial equations, Jacobian and cube quadrature checks."""
import unittest

from ml_collections import ConfigDict
import numpy as np

from examples.icemelting3d.extreme_reference import (
    evaluation_points, parameters, radial_system, sample_solution, solve_radial,
    weighted_metrics,
)


def config():
    return ConfigDict(dict(
        physical=dict(M=.1, lam=5., R0=35., eps_star=.00375),
        nondim=dict(L_c=100., T_c=5., xc=(.5, .5, .5)),
        experiment=dict(snapshots=3, quadrature_directions=512, quadrature_radial=256,
                        radial_points=256, radial_directions=16)))


class ExtremeReferenceTest(unittest.TestCase):
    def test_analytic_jacobian_matches_independent_directional_difference(self):
        radius = np.linspace(0., .9, 51)
        rhs, jac = radial_system(radius, *parameters(config()))
        u = .4*np.cos(2*np.pi*radius/.9)+.2*np.sin(3*radius)
        v = np.random.default_rng(11).normal(size=len(radius))
        step = 1e-6
        difference = (rhs(0., u+step*v)-rhs(0., u-step*v))/(2*step)
        np.testing.assert_allclose(jac(0., u)@v, difference, rtol=2e-8, atol=1e-8)

    def test_origin_and_reflected_outer_laplacian_have_second_order_consistency(self):
        errors = []
        a, b, g = parameters(config())
        for count in (65, 129):
            radius = np.linspace(0., .9, count)
            k = 2*np.pi/.9
            u = .3*np.cos(k*radius)
            gradient = -.3*k*np.sin(k*radius)
            exact_lap = -.3*k*k*np.cos(k*radius)
            exact_lap[1:] += 2*gradient[1:]/radius[1:]
            exact_lap[0] = -.9*k*k
            exact = a*(exact_lap-2*np.tanh(u)*gradient**2)+b*np.tanh(u)-g
            rhs, _ = radial_system(radius, a, b, g)
            errors.append(np.max(np.abs(rhs(0., u)-exact)))
        self.assertLess(errors[1], errors[0]/3.9)

    def test_cubature_covers_cube_with_correct_volume_measure(self):
        c = config()
        points, weights, rays, radius = evaluation_points(c)
        self.assertEqual(points.shape, (512*256, 3))
        self.assertEqual(rays.shape, (16, 256, 3))
        self.assertEqual(radius.shape, (256,))
        self.assertTrue(np.all((points > 0) & (points < 1)))
        self.assertAlmostEqual(weights.sum(), 1., delta=2e-4)
        # Exact cube integral of |x-center|**2 is1/4.
        self.assertAlmostEqual(np.dot(weights, np.sum((points-.5)**2, axis=1)), .25, delta=2e-4)
        changed = config()
        changed.physical.eps_star = .0009375
        for old, new in zip(evaluation_points(c), evaluation_points(changed), strict=True):
            np.testing.assert_array_equal(old, new)

    def test_radial_solve_refinement_and_exact_initial_profile(self):
        c = config()
        points = np.stack((np.linspace(.05, .95, 3001), np.full(3001, .5), np.full(3001, .5)), -1)
        solutions = [solve_radial(c, nodes=n, outer_radius=.9,
            max_step=.005, rtol=1e-9, atol=1e-10) for n in (1025, 2049)]
        fields = [sample_solution(s, points) for s in solutions]
        initial = np.tanh((.35-np.linalg.norm(points-.5, axis=1))/(np.sqrt(2)*c.physical.eps_star))
        np.testing.assert_allclose(fields[1][0], initial, atol=3e-14, rtol=1e-13)
        np.testing.assert_allclose(fields[0][1:], fields[1][1:], atol=3e-6, rtol=0)
        self.assertTrue(np.isfinite(solutions[1].latent).all())

    def test_weighted_metrics_exclude_initial_and_validate_inputs(self):
        truth = np.array([[1., 1.], [-.2, .3], [-.4, .5]])
        pred = truth.copy()
        pred[0] = -10
        self.assertEqual(weighted_metrics(pred, truth, np.array([.2, .8]))['pooled'], 0.)
        with self.assertRaises(ValueError):
            weighted_metrics(pred, truth, np.array([-.2, .8]))
        with self.assertRaises(ValueError):
            solve_radial(config(), nodes=256, outer_radius=.5, max_step=.01, rtol=1e-8, atol=1e-9)
