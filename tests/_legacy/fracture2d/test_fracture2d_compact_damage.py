"""Independent compact-feature values, normalization and parameter gradient."""
import unittest
from types import SimpleNamespace
import jax
import jax.numpy as jnp
import numpy as np
from scipy.sparse import csr_matrix, diags
from examples.fracture2d._legacy.compact_damage_features import (
    Readout, build_basis, pack_basis, residual_column_scale, residual_power_loss,
)
from examples.fracture2d._legacy.train import _nonnegative_damage


class CompactDamageTest(unittest.TestCase):
    def test_positive_baseline_restores_link_gradient(self):
        jax.config.update('jax_enable_x64', True)
        floor = 1e-8
        raw = jnp.asarray(-.01)
        baseline = _nonnegative_damage(raw, floor)
        derivative = jax.grad(lambda delta: _nonnegative_damage(baseline+delta, floor))(jnp.asarray(0.))
        self.assertGreater(float(derivative), .49)
        self.assertLess(float(abs(_nonnegative_damage(baseline, floor)-baseline)), floor)

    def test_basis_normalization_and_gradient(self):
        jax.config.update('jax_enable_x64', True)
        axis = np.linspace(0, 1, 21)
        x, y = np.meshgrid(axis, axis)
        points = np.column_stack((x.ravel(), y.ravel()))
        residual = points[:, 0]+points[:, 1]
        controls = SimpleNamespace(grid_step=.2, coarse_radius=.3,
            adaptive_centers=8, minimum_separation=.01,
            minimum_radius=.08, maximum_radius=.12, radius_factor=2.)
        basis, centers, radii, selected = build_basis(points, residual, controls)
        independent = np.maximum(1-np.linalg.norm(
            points[:, None, :]-centers[None, :, :], axis=2)/radii, 0)
        np.testing.assert_allclose(basis.toarray(), independent, atol=1e-14)
        np.testing.assert_array_equal(points[selected], centers)
        # Largest-residual corner already has a coarse unit: a fine unit must
        # still be selectable there, with a distinct support radius.
        self.assertEqual(np.count_nonzero(selected == np.argmax(residual)), 2)
        controls.radius_factor = 1.2
        controls.minimum_radius = .04
        nested, _, _, _ = build_basis(points, residual, controls, inherited=(selected, radii))
        np.testing.assert_array_equal(nested[:, :basis.shape[1]].toarray(), basis.toarray())
        controls.adaptive_centers = 0
        unchanged, _, _, _ = build_basis(points, residual, controls, inherited=(selected, radii))
        np.testing.assert_array_equal(unchanged.toarray(), basis.toarray())
        n = len(points)
        D = diags([-np.ones(n-1), np.r_[1., np.full(n-2, 2.), 1.], -np.ones(n-1)], [-1, 0, 1]).tocsr()
        H = 1+points[:, 0]
        scale = residual_column_scale(D, basis, H, gc0=2., gc1=.3)
        action = (diags(2+2*H)@basis+.3*D@basis).toarray()*scale
        np.testing.assert_allclose(np.mean((action/(2*np.sqrt(np.mean(H**2))))**2, axis=0), 1., rtol=1e-13)
        field, width = pack_basis(basis, scale)
        coefficients = jnp.linspace(-.01, .02, basis.shape[1])
        np.testing.assert_allclose(field(Readout(coefficients)), independent@(scale*np.asarray(coefficients)), atol=1e-14)
        A = jnp.asarray((diags(2+2*H)+.3*D).toarray())
        def loss(beta):
            return jnp.mean((A@field(Readout(beta))-2*jnp.asarray(H))**2)
        direction = jnp.sin(jnp.arange(basis.shape[1]))
        analytic = jnp.dot(jax.grad(loss)(coefficients), direction)
        eps = 1e-5
        fd = (loss(coefficients+eps*direction)-loss(coefficients-eps*direction))/(2*eps)
        np.testing.assert_allclose(fd, analytic, rtol=1e-8, atol=1e-8)
        for power in (4, 16, 32):
            def peak(beta):
                rho = A@field(Readout(beta))-2*jnp.asarray(H)
                return residual_power_loss(rho, jnp.asarray(H), power=power, floor=1e-15)
            analytic = jnp.dot(jax.grad(peak)(coefficients), direction)
            fd = (peak(coefficients+eps*direction)-peak(coefficients-eps*direction))/(2*eps)
            np.testing.assert_allclose(fd, analytic, rtol=1e-7, atol=1e-8)
        zero = jnp.zeros(3)
        value = residual_power_loss(zero, jnp.ones(3), power=32, floor=1e-15)
        np.testing.assert_allclose(value, 1e-30, rtol=1e-13, atol=0)
        self.assertTrue(np.isfinite(np.asarray(jax.grad(lambda r: residual_power_loss(r, jnp.ones(3), power=32, floor=1e-15))(zero))).all())
        self.assertGreater(width, 0)


if __name__ == '__main__':
    unittest.main()
