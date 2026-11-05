"""Independent derivative and physical-boundary checks for the local AD control."""
import unittest

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from examples.heterogeneous_corrosion2d.configs.train_base import get_config
from examples.heterogeneous_corrosion2d.local_ad import (
    encoded_raw, local_residual, make_loss_vector, normal_derivatives,
    phase_driving_fields, spatial_laplacians,
)
from examples.heterogeneous_corrosion2d.network import build_network, physical_fields
from examples.heterogeneous_corrosion2d.problem import initial_fields, nondim
from examples.heterogeneous_corrosion2d.boundaries import neumann_quadrature


class PolynomialOperator(eqx.Module):
    """Known physical polynomials, represented through the actual output map."""
    amplitude: jax.Array

    def encode(self, image):
        return self.amplitude*jnp.mean(image)

    def prepare_encoded(self, encoded, points, local):
        if local.shape != (len(points), 0):
            raise ValueError('This manufactured model requires no local feature')
        return jnp.concatenate((points, jnp.full((len(points), 1), encoded)), axis=1)

    def decode(self, prepared, times):
        x, y, condition = [prepared[None, :, i] for i in range(3)]
        t = times[:, None]
        phi = .25+.1*x*x+.07*x*y+.08*y*y+.03*t+.02*condition
        liquid = 1+.12*x**3-.04*x*y+.08*y*y+.06*t+.03*t*x+.1*condition
        return jnp.stack((jnp.arctanh(2*phi-1), liquid), axis=-1)


def tiny_config():
    c = get_config()
    c.geometry.length = 1.7
    c.sample.t_max = 2.3
    c.loss.neumann_scales = (1., 1.)
    c.operator.image_shape = (1, 2, 4)
    c.arch.patch_shape = (1, 2)
    c.arch.width, c.arch.heads = 8, 2
    c.arch.encoder_depth, c.arch.decoder_depth = 1, 1
    c.arch.spatial_features, c.arch.time_features = 2, 1
    c.arch.extent = (c.geometry.length, c.geometry.height)
    return c


class LocalDerivativeTests(unittest.TestCase):
    def fixture(self):
        c = tiny_config()
        model = PolynomialOperator(jnp.asarray(.7))
        image = jnp.linspace(-.7, .9, 8).reshape(1, 2, 4)
        points = jnp.array([[.2, .1], [.65, .2], [.8, .4]])
        times = jnp.array([.14, .62])
        return c, model, image, points, times

    def test_analytic_polynomial_laplacian_normal_and_physical_time(self):
        with jax.enable_x64(True):
            c, model, image, points, times = self.fixture()
            encoded = model.encode(image)
            x, y = np.asarray(points).T
            tau = np.asarray(times)[:, None]/c.sample.t_max
            actual = np.asarray(spatial_laplacians(model, encoded, points, times, c))
            expected = np.broadcast_to(np.stack((np.full(len(points), .36),
                c.physical.c_le*(.72*x+.16)), axis=-1), actual.shape)
            np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=2e-14)
            normals = jnp.array([[-1., 0.], [0., 1.], [1., 0.]])
            actual_normal = normal_derivatives(model, encoded, points, normals, times, c)
            grad_phi = np.stack((.2*x+.07*y, .07*x+.16*y), axis=-1)
            grad_q = np.stack((np.broadcast_to(.36*x*x-.04*y, tau.shape[:1]+x.shape)+.03*tau,
                               np.broadcast_to(-.04*x+.16*y, tau.shape[:1]+x.shape)), axis=-1)
            expected_normal = c.geometry.length*np.stack((
                np.broadcast_to(np.sum(grad_phi*np.asarray(normals), axis=-1), tau.shape[:1]+x.shape),
                np.sum(grad_q*np.asarray(normals)[None], axis=-1)), axis=-1)
            np.testing.assert_allclose(actual_normal, expected_normal, rtol=2e-13, atol=2e-14)

            mobility = jnp.array([1.1, 2.8, 1.5])
            ac, ch = local_residual(model, encoded, points, times, mobility, c)
            raw = encoded_raw(model, encoded, points, times, c)
            phi, concentration = np.asarray(physical_fields(raw, c)).transpose(2, 0, 1)
            nch, n1, n2, n3, dc, cle = nondim(c)
            q = concentration-dc*phi**2*(3-2*phi)-cle
            hp, gp = 6*phi*(1-phi), 2*phi*(1-phi)*(1-2*phi)
            dp = np.full_like(phi, .03/c.sample.t_max)
            dtc = cle*(.06+.03*x)/c.sample.t_max+dc*hp*dp
            expected_ac = dp+np.asarray(mobility)[None]*(-n3*.36+n2*gp-n1*q*dc*hp)
            expected_ch = dtc-nch*cle*(.72*x+.16)
            np.testing.assert_allclose(ac, expected_ac, rtol=2e-12, atol=2e-13)
            np.testing.assert_allclose(ch, expected_ch, rtol=2e-12, atol=2e-13)
            ac_one, ch_one = local_residual(model, encoded, points, times, jnp.ones_like(mobility), c)
            np.testing.assert_allclose(ac-dp, np.asarray(mobility)[None]*(ac_one-dp), atol=1e-13)
            np.testing.assert_array_equal(ch, ch_one)

    def test_batched_directional_laplacian_matches_pointwise_network_hessian(self):
        with jax.enable_x64(True):
            c = tiny_config()
            model = build_network(c, jax.random.key(91))
            image = jnp.linspace(-.8, .7, 8).reshape(1, 2, 4)
            points = jnp.array([[.21, .13], [.75, .41]])
            times = jnp.array([.17, .71])
            encoded = model.encode(image)
            actual = spatial_laplacians(model, encoded, points, times, c)

            def point(x):
                return phase_driving_fields(encoded_raw(model, encoded, x[None], times, c), c)[:, 0]

            hessian = jax.vmap(jax.jacfwd(jax.jacrev(point)))(points)
            expected = jnp.trace(hessian, axis1=-2, axis2=-1).transpose(1, 0, 2)
            np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-12)
            normals = jnp.array([[-1., 0.], [0., 1.]])
            jacobian = jax.vmap(jax.jacfwd(point))(points)
            expected_normal = jnp.einsum('ntci,ni->tnc', jacobian, normals)
            expected_normal = expected_normal*c.geometry.length/jnp.array([1., c.physical.c_le])
            np.testing.assert_allclose(normal_derivatives(model, encoded, points, normals, times, c),
                                       expected_normal, rtol=1e-11, atol=1e-12)
            # This explicitly checks the architectural condition that makes a
            # common directional tangent correct: other query rows do not enter.
            full = jax.jacfwd(lambda x: phase_driving_fields(
                encoded_raw(model, encoded, x, times, c), c))(points)
            np.testing.assert_array_equal(full[:, 0, :, 1], 0)
            np.testing.assert_array_equal(full[:, 1, :, 0], 0)


class LocalLossTests(unittest.TestCase):
    def fixture(self):
        c = tiny_config()
        model = PolynomialOperator(jnp.asarray(.7))
        images = jnp.linspace(-.9, .8, 16).reshape(2, 1, 2, 4)
        # Three cores, two IC points, one mouth, three wall quadrature points.
        points = jnp.array([[.2, .1], [.65, .2], [.8, .4], [.3, .2], [.5, .1],
                            [.5, 0.], [0., .2], [1.7, .4], [.8, .6]])
        normals = np.array([[-1., 0.], [1., 0.], [0., 1.]])
        weights = np.array([.2, .4, .7])
        initial = jnp.asarray(initial_fields(c, np.asarray(points[3:5])))
        kappa = jnp.array([[1.1, 2.8, 1.5], [2.2, 1.3, 2.9]])
        mask, times = jnp.array([True, False, True]), jnp.array([.14, .62])
        return c, model, times, (images, kappa, points, initial), mask, normals, weights

    def test_seven_loss_reductions_and_length_weighted_neumann(self):
        with jax.enable_x64(True):
            c, model, times, payload, mask, normals, weights = self.fixture()
            images, kappa, points, initial = payload
            loss = make_loss_vector(c, 3, 2, 1, 3, normals, weights)
            actual = loss(model, times, payload, None, mask)
            expected = []
            for image, mobility in zip(images, kappa, strict=True):
                encoded = model.encode(image)
                ac, ch = local_residual(model, encoded, points[:3], times, mobility, c)
                field = physical_fields(encoded_raw(model, encoded, points, times, c), c)
                zero = physical_fields(encoded_raw(model, encoded, points[3:5], jnp.zeros(1), c), c)[0]
                normal = normal_derivatives(model, encoded, points[-3:], jnp.asarray(normals), times, c)
                # Explicit scalar summation independently checks the quadrature
                # normalization and the separate phase/concentration BC terms.
                bn = [sum(float(normal[t, p, f])**2*weights[p]
                          for t in range(2) for p in range(3))/(2*weights.sum()) for f in range(2)]
                expected.append([c.loss.ac_scale*np.abs(np.asarray(ac)[:, [0, 2]]).mean(),
                    c.loss.ch_scale*np.abs(np.asarray(ch)[:, [0, 2]]).mean(),
                    np.mean(np.asarray(field[:, 5, 0])**2),
                    np.mean((np.asarray(field[:, 5, 1])/c.physical.c_le)**2),
                    np.mean(np.sum(np.asarray((zero-initial)/jnp.array([1., c.physical.c_le]))**2, axis=-1)),
                    *bn])
            np.testing.assert_allclose(actual, np.mean(expected, axis=0), rtol=1e-12, atol=1e-12)
            # Scaling a quadrature measure cannot change its normalized loss.
            scaled = make_loss_vector(c, 3, 2, 1, 3, normals, 7*weights)
            np.testing.assert_allclose(actual, scaled(model, times, payload, None, mask), atol=1e-12)
            changed_ic = loss(model, times, (images, kappa, points, jnp.zeros_like(initial)), None, mask)
            np.testing.assert_array_equal(actual[jnp.array([0, 1, 2, 3, 5, 6])],
                                          changed_ic[jnp.array([0, 1, 2, 3, 5, 6])])

    def test_encoder_parameter_gradient_through_spatial_ad_matches_finite_difference(self):
        with jax.enable_x64(True):
            c, model, times, payload, mask, normals, weights = self.fixture()
            loss = make_loss_vector(c, 3, 2, 1, 3, normals, weights)

            def objective(amplitude):
                candidate = eqx.tree_at(lambda m: m.amplitude, model, amplitude)
                return loss(candidate, times, payload, None, mask).sum()

            actual = jax.grad(objective)(model.amplitude)
            eps = 1e-5
            finite = (objective(model.amplitude+eps)-objective(model.amplitude-eps))/(2*eps)
            self.assertGreater(abs(float(actual)), 1e-4)
            np.testing.assert_allclose(actual, finite, rtol=2e-7, atol=2e-8)

    def test_invalid_shapes_normals_weights_and_nonlocal_payload_fail_loudly(self):
        c, model, times, payload, mask, normals, weights = self.fixture()
        for bad_normals, bad_weights in ((normals[:-1], weights), (2*normals, weights),
                                          (normals, weights[:-1]), (normals, -weights),
                                          (normals, np.array([1., np.nan, 1.]))):
            with self.subTest(normals=bad_normals, weights=bad_weights), self.assertRaises(ValueError):
                make_loss_vector(c, 3, 2, 1, 3, bad_normals, bad_weights)
        loss = make_loss_vector(c, 3, 2, 1, 3, normals, weights)
        with self.assertRaisesRegex(ValueError, 'None'):
            loss(model, times, payload, (), mask)
        with self.assertRaisesRegex(ValueError, 'shape'):
            loss(model, times, payload, None, mask[:-1])
        c.loss.neumann_scales = (1.,)
        with self.assertRaisesRegex(ValueError, 'two'):
            make_loss_vector(c, 3, 2, 1, 3, normals, weights)


class NeumannQuadratureTests(unittest.TestCase):
    def test_default_five_open_segments_counts_measure_and_normals(self):
        c = get_config()
        points, normals, weights = neumann_quadrature(c)
        self.assertEqual(points.shape, (248, 2))
        for array in (points, normals, weights):
            self.assertEqual(array.dtype, np.float64)
        lengths = [.6, .6, 1., .445, .445]
        counts = [48, 48, 80, 36, 36]
        expected_normals = [(-1., 0.), (1., 0.), (0., 1.), (0., -1.), (0., -1.)]
        offset = 0
        for length, count, normal in zip(lengths, counts, expected_normals, strict=True):
            np.testing.assert_array_equal(normals[offset:offset+count],
                                          np.broadcast_to(normal, (count, 2)))
            np.testing.assert_allclose(weights[offset:offset+count], length/count, atol=1e-16)
            offset += count
        self.assertAlmostEqual(weights.sum(), 3.09, places=14)
        self.assertTrue(np.all(weights > 0))
        self.assertTrue(np.all((points[:, 0] >= 0) & (points[:, 0] <= c.geometry.length)))
        self.assertTrue(np.all((points[:, 1] >= 0) & (points[:, 1] <= c.geometry.height)))
        on_top = points[:, 1] == 0
        self.assertTrue(np.all(np.abs(points[on_top, 0]-c.geometry.pit_x) > c.geometry.mouth_half_width))
        endpoints = np.array([[0., 0.], [1., 0.], [0., .6], [1., .6], [.445, 0.], [.555, 0.]])
        self.assertGreater(float(np.linalg.norm(points[:, None]-endpoints[None], axis=-1).min()), 0.)
        self.assertEqual(len(np.unique(points, axis=0)), len(points))
        # Constant and linear line functions are integrated exactly by midpoint
        # quadrature, including the asymmetry of an off-centre reservoir mouth.
        c.geometry.pit_x = .41
        points, _, weights = neumann_quadrature(c)
        a, b = c.geometry.pit_x-c.geometry.mouth_half_width, c.geometry.pit_x+c.geometry.mouth_half_width
        exact_x = (c.geometry.length*c.geometry.height+c.geometry.length**2
                   -(b*b-a*a)/2)
        exact_y = c.geometry.height**2+c.geometry.length*c.geometry.height
        np.testing.assert_allclose(weights@points, [exact_x, exact_y], atol=1e-15)

    def test_integer_ratio_roundoff_and_invalid_geometry(self):
        c = get_config()
        c.geometry.height = np.nextafter(c.geometry.height, np.inf)
        self.assertEqual(len(neumann_quadrature(c)[0]), 248)
        for center, half in ((.01, .055), (.99, .055), (.5, 0.), (.5, .5), (.5, np.nan)):
            broken = get_config()
            broken.geometry.pit_x, broken.geometry.mouth_half_width = center, half
            with self.subTest(center=center, half=half), self.assertRaisesRegex(ValueError, 'mouth'):
                neumann_quadrature(broken)
        broken = get_config()
        broken.sample.Nx = 1
        with self.assertRaisesRegex(ValueError, 'Nx'):
            neumann_quadrature(broken)


if __name__ == '__main__':
    unittest.main()
