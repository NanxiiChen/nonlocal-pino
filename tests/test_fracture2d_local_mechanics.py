"""Local FEM weak-form checks for the isolated fracture trial."""
import os
import unittest
import numpy as np
from scipy.sparse.linalg import spsolve
from examples.fracture2d.local_mechanics import LocalMechanics


def model():
    points = np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]])
    triangles = np.array([[0, 1, 2], [1, 3, 2]], np.int64)
    volumes = np.array([1/6, 1/3, 1/3, 1/6])
    direction = np.array([0., 0., 0., 0., 0., 1., 0., 1.])
    return LocalMechanics(points, triangles, 4, 2, volumes, E=210000., nu=.3,
        k_res=1e-8, constrained=np.array([0, 1, 3, 5, 7], np.int64), load_direction=direction, range_atol=1e-3)


class LocalMechanicsTest(unittest.TestCase):
    def test_plane_strain_affine_patch_and_reaction(self):
        m = model()
        load = 1e-4
        q = np.column_stack((-m.coordinates[:, 0]*load*.3/.7, m.coordinates[:, 1]*load)).ravel()
        K = m.matrix(np.zeros(4))
        np.testing.assert_allclose((K @ q)[m.free], 0., atol=1e-13)
        self.assertAlmostEqual(float((K @ q) @ m.load_direction), 210000/(1-.3**2)*load*(1+1e-8), places=11)
        np.testing.assert_allclose(m.driver(q), m.driver(q)[0], rtol=1e-14)
        self.assertGreater(m.driver(q)[0], 0)

    def test_weak_form_energy_and_degree_two_degradation(self):
        m = model()
        d = np.array([.1, .4, .7, .9])
        values = d[m.triangles]
        expected = 1-2*values.mean(axis=1)+(np.sum(values, axis=1)**2+np.sum(values**2, axis=1))/12+1e-8
        np.testing.assert_allclose(m.factors(d), expected, rtol=1e-14)
        K = m.matrix(d)
        np.testing.assert_allclose(K.toarray(), K.T.toarray(), atol=1e-11)
        rng = np.random.default_rng(1)
        q, v = rng.normal(size=(2, 8))*1e-4
        eps = 1e-3
        fd = (.5*(q+eps*v) @ K @ (q+eps*v)-.5*(q-eps*v) @ K @ (q-eps*v))/(2*eps)
        self.assertAlmostEqual(fd, v @ K @ q, places=13)

    def test_rigid_fields_and_invalid_inputs(self):
        m = model()
        translation = np.tile([.1, -.2], 4)
        rotation = np.column_stack((-m.coordinates[:, 1], m.coordinates[:, 0])).ravel()
        for q in (translation, rotation):
            np.testing.assert_allclose(m.matrix(np.zeros(4)) @ q, 0., atol=1e-10)
            np.testing.assert_allclose(m.driver(q), 0., atol=1e-24)
        with self.assertRaises(ValueError):
            m.factors(np.array([0., 0., 0., np.nan]))
        with self.assertRaises(ValueError):
            m.factors(np.full(4, 1.1))

    @unittest.skipUnless(os.environ.get('RUN_FRACTURE2D_GPU_TESTS') == '1', 'explicit serialized GPU test')
    def test_gpu_matches_host_solve_and_rejects_truncation(self):
        import jax
        jax.config.update('jax_enable_x64', True)
        if jax.devices()[0].platform != 'gpu':
            self.skipTest('GPU unavailable')
        m = model()
        d, load = np.array([.05, .15, .2, .4]), 1e-4
        K = m.matrix(d)
        q_expected = load*m.load_direction
        q_expected[m.free] = spsolve(K[m.free][:, m.free], -(K @ q_expected)[m.free])
        solve = m.gpu_solver(tol=1e-10, atol=1e-12, maxiter=100)
        q, reaction, info = solve(d, load, np.zeros(8))
        np.testing.assert_allclose(q, q_expected, rtol=1e-9, atol=1e-13)
        self.assertAlmostEqual(reaction, (K @ q_expected) @ m.load_direction, places=9)
        self.assertTrue(info['converged'])
        with self.assertRaises(RuntimeError):
            m.gpu_solver(tol=1e-13, atol=1e-15, maxiter=1)(d, load, np.zeros(8))


if __name__ == '__main__':
    unittest.main()
