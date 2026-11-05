"""Independent mobility, conservation and conditional-physics checks on CPU."""
import unittest
import tempfile
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from scipy import sparse

from examples.heterogeneous_corrosion2d.configs.train_base import get_config
from examples.heterogeneous_corrosion2d import problem
from examples.heterogeneous_corrosion2d.solver import make_system
from examples.heterogeneous_corrosion2d.network import build_network, predict_fields
from examples.heterogeneous_corrosion2d.losses import graph_action, make_loss_vector
from examples.heterogeneous_corrosion2d.evaluate import load_reference


def small_config():
    c = get_config()
    c.sample.Nx, c.sample.Ny = 9, 7
    c.operator.image_shape = (1, 4, 8)
    c.arch.patch_shape = (2, 2)
    c.arch.width = 8
    c.arch.heads = 2
    c.arch.encoder_depth = 1
    c.arch.decoder_depth = 2
    c.arch.spatial_features = 3
    c.arch.time_features = 2
    return c


class HeterogeneousPhysicsTests(unittest.TestCase):
    def test_local_operator_sign_moments_and_closed_conservation(self):
        c = small_config()
        xx, yy, _ = problem.grid(c)
        matrix = problem.local_matrix(c)
        weights = problem.quadrature(c)
        np.testing.assert_allclose(matrix @ np.ones(xx.size), 0, atol=1e-12)
        polynomial = (xx**2 + yy**2).ravel()
        action = (matrix @ polynomial).reshape(xx.shape)
        np.testing.assert_allclose(action[1:-1, 1:-1], -4, atol=1e-12)
        weighted = sparse.diags(weights) @ matrix
        np.testing.assert_allclose((weighted - weighted.T).toarray(), 0, atol=1e-14)
        q = np.random.default_rng(12).normal(size=xx.size)
        self.assertGreater(float(q @ weighted @ q), 0)
        self.assertAlmostEqual(float(weights @ (matrix @ q)), 0, places=12)
        self.assertAlmostEqual(float(weights.sum()), c.geometry.length*c.geometry.height)

    def test_material_image_and_node_values_share_one_continuous_input(self):
        c = small_config()
        parameters = problem.sample_parameters(c, np.random.default_rng(4), 5)
        points = problem.image_points(c)
        actual = problem.material_field(c, parameters, points, np)
        modes = np.array(c.material.modes)
        phase = 2*np.pi*(points/np.array([c.geometry.length, c.geometry.height])) @ modes.T
        raw = parameters[:, :1] + (
            parameters[:, 1:1+len(modes)] @ np.cos(phase).T
            + parameters[:, 1+len(modes):] @ np.sin(phase).T
        ) / np.sqrt(len(modes))
        expected = c.material.minimum + (c.material.maximum-c.material.minimum)/(1+np.exp(-raw))
        np.testing.assert_allclose(actual, expected, atol=1e-14, rtol=1e-14)
        self.assertGreaterEqual(actual.min(), c.material.minimum)
        self.assertLessEqual(actual.max(), c.material.maximum)
        self.assertGreater(np.linalg.norm(actual[0]-actual[1]), 0.1)
        normalized = 2*(actual-c.material.minimum)/(c.material.maximum-c.material.minimum)-1
        np.testing.assert_array_equal(problem.material_image(c, parameters, np).reshape(5, -1), normalized)
        with jax.enable_x64(True):
            np.testing.assert_allclose(
                problem.material_image(c, jnp.asarray(parameters), jnp),
                normalized.reshape((5, *c.operator.image_shape)), atol=1e-14, rtol=1e-14)
        with self.assertRaisesRegex(ValueError, 'shape'):
            problem.material_field(c, parameters[:, :-1], points, np)

    def test_stronger_prior_fixes_only_constant_mode_and_preserves_spatial_draws(self):
        from examples.heterogeneous_corrosion2d.configs.train_stronger import get_config as stronger_config
        baseline, stronger = get_config(), stronger_config()
        problem.validate(baseline)
        problem.validate(stronger)
        count, seed = 8, 7621
        baseline_rng, stronger_rng = np.random.default_rng(seed), np.random.default_rng(seed)
        a = problem.sample_parameters(baseline, baseline_rng, count)
        b = problem.sample_parameters(stronger, stronger_rng, count)
        # Existing scalar sampling is bitwise unchanged, including RNG use.
        expected = np.random.default_rng(seed).normal(0., baseline.material.coefficient_scale, a.shape)
        np.testing.assert_array_equal(a, expected)
        np.testing.assert_array_equal(b[:, 0], 0.)
        scales = np.asarray(stronger.material.coefficient_scale)
        np.testing.assert_allclose(b[:, 1:]/scales[1:],
                                   a[:, 1:]/baseline.material.coefficient_scale,
                                   rtol=2e-15, atol=2e-15)
        self.assertEqual(baseline_rng.random(), stronger_rng.random())
        points = problem.image_points(stronger)
        fields = problem.material_field(stronger, b, points, np)
        self.assertTrue(np.isfinite(fields).all())
        self.assertGreaterEqual(float(fields.min()), stronger.material.minimum)
        self.assertLessEqual(float(fields.max()), stronger.material.maximum)
        # Pixel-centre quadrature resolves all integer modes. Fixing the latent
        # mean does not assert a fixed physical mean after the sigmoid.
        modes = np.asarray(stronger.material.modes)
        phase = 2*np.pi*(points/np.array([stronger.geometry.length, stronger.geometry.height])) @ modes.T
        latent = (b[:, 1:1+len(modes)] @ np.cos(phase).T
                  + b[:, 1+len(modes):] @ np.sin(phase).T)/np.sqrt(len(modes))
        np.testing.assert_allclose(latent.mean(axis=1), 0., atol=1e-14)
        self.assertGreater(np.ptp(fields.mean(axis=1)), 1e-3)
        size = problem.parameter_count(stronger)
        for invalid in ((0., -1.)+(1.,)*(size-2), (1.,)*(size-1),
                        (1.,)+(0.,)*(size-1), (0.,)*size, -0.5, 0., float('nan')):
            broken = stronger.copy_and_resolve_references()
            with broken.material.ignore_type():
                broken.material.coefficient_scale = invalid
            with self.subTest(scale=invalid), self.assertRaisesRegex(ValueError, 'coefficient_scale'):
                problem.validate(broken)

    def test_initial_width_and_mouth_are_consistent(self):
        c = small_config()
        _, _, points = problem.grid(c)
        fields = problem.initial_fields(c, points)
        mouth = problem.boundary_mask(c, points)
        self.assertTrue(mouth.any())
        np.testing.assert_array_equal(fields[mouth], 0)
        phi = fields[:, 0]
        np.testing.assert_allclose(fields[:, 1], c.physical.c_se*(3*phi**2-2*phi**3))
        # A planar profile has zero double-well/gradient chemical potential.
        x = np.linspace(-0.1, 0.1, 17)
        p = 0.5*(1+np.tanh(c.geometry.interface_k*x))
        d2p = 4*c.geometry.interface_k**2*p*(1-p)*(1-2*p)
        residual = -c.physical.n_ac3*d2p + c.physical.n_ac2*2*p*(1-p)*(1-2*p)
        np.testing.assert_allclose(residual, 0, atol=1e-14)
        c.geometry.interface_k *= 1.1
        with self.assertRaisesRegex(ValueError, 'width'):
            problem.validate(c)

    def test_sparse_analytic_jacobian_matches_independent_directional_difference(self):
        c = small_config()
        _, _, points = problem.grid(c)
        matrix = problem.local_matrix(c)
        kappa = np.linspace(1, 3, len(points))
        rhs, jacobian, state, _ = make_system(c, matrix, points, kappa)
        rng = np.random.default_rng(19)
        nf = len(state)//2
        state = np.r_[rng.uniform(0.15, 0.85, nf), rng.uniform(0.05, 0.95, nf)]
        direction = rng.normal(size=len(state))
        direction /= np.linalg.norm(direction)
        step = 1e-6
        actual = jacobian(0.27, state) @ direction
        finite = (rhs(0.27, state+step*direction)-rhs(0.27, state-step*direction))/(2*step)
        np.testing.assert_allclose(actual, finite, rtol=2e-7, atol=2e-6)

    def test_reference_rejects_reversed_operator_and_inert_material(self):
        c = small_config()
        _, _, points = problem.grid(c)
        matrix = problem.local_matrix(c)
        with self.assertRaisesRegex(ValueError, 'positive|negative|sign'):
            make_system(c, -matrix, points, np.ones(len(points)))
        kappa = np.ones(len(points))
        kappa[2] = 0
        with self.assertRaisesRegex(ValueError, 'positive'):
            make_system(c, matrix, points, kappa)

    def test_material_mobility_is_variational_ac_only_and_reservoir_flux_balances(self):
        c = small_config()
        _, _, points = problem.grid(c)
        matrix = problem.local_matrix(c)
        weights = problem.quadrature(c)
        mouth = problem.boundary_mask(c, points)
        free = ~mouth
        kappa = np.linspace(1, 3, len(points))
        rhs, _, state, unpack = make_system(c, matrix, points, kappa)
        constant_rhs, _, _, _ = make_system(c, matrix, points, np.ones(len(points)))
        rng = np.random.default_rng(21)
        nf = free.sum()
        state = np.r_[rng.uniform(0.1, 0.9, nf), rng.uniform(0.05, 0.9, nf)]
        rate = rhs(0.3, state)
        constant_rate = constant_rhs(0.3, state)
        np.testing.assert_allclose(rate[:nf], kappa[free]*constant_rate[:nf], atol=1e-13)
        np.testing.assert_array_equal(rate[nf:], constant_rate[nf:])
        nch, n1, n2, n3, dc, cle = problem.nondim(c)

        def energy(values):
            p, concentration = unpack(values).T
            q = concentration-dc*(3*p*p-2*p*p*p)-cle
            return (n3/2*p @ (weights*(matrix @ p))
                    + weights @ (n2*p*p*(1-p)**2+n1/2*q*q))

        tangent = rng.normal(size=nf)
        tangent /= np.linalg.norm(tangent)
        direction = np.r_[tangent, np.zeros(nf)]
        step = 1e-6
        finite_energy = (energy(state+step*direction)-energy(state-step*direction))/(2*step)
        dissipation_pair = -np.dot(weights[free]*rate[:nf]/kappa[free], tangent)
        np.testing.assert_allclose(finite_energy, dissipation_pair, rtol=1e-7, atol=1e-9)

        p, concentration = unpack(state).T
        q = concentration-dc*(3*p*p-2*p*p*p)-cle
        # Reflecting faces contribute zero; only free-to-reservoir faces remain.
        links = matrix[free][:, mouth].tocoo()
        free_ids, mouth_ids = np.flatnonzero(free), np.flatnonzero(mouth)
        face_flux = -nch*np.sum(weights[free_ids[links.row]]*links.data
                               *(q[mouth_ids[links.col]]-q[free_ids[links.row]]))
        np.testing.assert_allclose(weights[free] @ rate[nf:], face_flux, atol=1e-12)


class HeterogeneousNetworkTests(unittest.TestCase):
    def fixture(self):
        c = small_config()
        c.sample.t_max = 2.3
        net = build_network(c, jax.random.key(33))
        points = jnp.array([[0.2, 0.1], [0.65, 0.2], [0.8, 0.4],
                            [0.45, 0.1], [0.5, 0.25], [0.5, 0.]])
        images = jnp.linspace(-0.9, 0.8, 64).reshape(2, 1, 4, 8)
        kappa = jnp.array([[1.1, 2.8, 1.5], [2.2, 1.3, 2.9]])
        initial = jnp.asarray(problem.initial_fields(c, np.asarray(points[3:5])))
        row, col = jnp.array([0, 0, 1, 2]), jnp.array([1, 2, 0, 0])
        sub = row, col, jnp.array([-2., -1., -2., -1.]), jnp.array([3., 2., 1.])
        mask, times = jnp.array([True, False, True]), jnp.array([0.14, 0.62])
        return c, net, times, (images, kappa, points, initial), sub, mask

    def test_time_jvp_has_physical_chain_rule_and_no_query_time_coupling(self):
        with jax.enable_x64(True):
            c, net, times, payload, _, _ = self.fixture()
            images, _, points, _ = payload
            fn = lambda ts: predict_fields(net, images[0], points, ts, c)
            _, actual = jax.jvp(fn, (times,), (jnp.ones_like(times),))
            step = 1e-5
            finite = (fn(times+step)-fn(times-step))/(2*step)
            np.testing.assert_allclose(actual, finite, rtol=2e-6, atol=1e-9)
            jacobian = jax.jacfwd(fn)(times)
            np.testing.assert_array_equal(jacobian[0, :, :, 1], 0)
            np.testing.assert_array_equal(jacobian[1, :, :, 0], 0)
            self.assertEqual(net.local_dim, 0)
            self.assertGreater(float(jnp.linalg.norm(fn(times)-predict_fields(
                net, images[1], points, times, c))), 1e-4)

    def test_five_losses_match_dense_physical_oracle_without_future_labels(self):
        with jax.enable_x64(True):
            c, net, times, payload, sub, mask = self.fixture()
            images, kappa, points, initial = payload
            fn = make_loss_vector(c, 3, 2, 1)
            actual = fn(net, times, payload, sub, mask)
            matrix = np.array([[3., -2., -1.], [-2., 2., 0.], [-1., 0., 1.]])
            expected = []
            nch, n1, n2, n3, dc, cle = problem.nondim(c)
            for image, mobility in zip(images, kappa, strict=True):
                fields = lambda ts: predict_fields(net, image, points, ts, c)
                values = np.asarray(fields(times))
                step = 1e-5
                dt = np.asarray((fields(times+step)-fields(times-step))/(2*step))
                p, concentration = values[:, :3, 0].T, values[:, :3, 1].T
                q = concentration-dc*(3*p*p-2*p*p*p)-cle
                ac = dt[:, :3, 0].T + np.asarray(mobility)[:, None]*(
                    n3*matrix @ p + n2*2*p*(1-p)*(1-2*p)-n1*q*dc*6*p*(1-p))
                ch = dt[:, :3, 1].T + nch*matrix @ q
                ic = np.asarray(fields(jnp.zeros(1)))[0, 3:5]
                expected.append([
                    c.loss.ac_scale*np.abs(ac[[0, 2]]).mean(),
                    c.loss.ch_scale*np.abs(ch[[0, 2]]).mean(),
                    np.mean(values[:, 5:, 0]**2),
                    np.mean((values[:, 5:, 1]/cle)**2),
                    np.mean(np.sum(((ic-initial)/np.array([1., cle]))**2, axis=-1))])
            np.testing.assert_allclose(actual, np.mean(expected, axis=0), rtol=2e-7, atol=2e-8)
            altered = fn(net, times, (images, kappa, points, jnp.zeros_like(initial)), sub, mask)
            np.testing.assert_array_equal(actual[:4], altered[:4])
            self.assertGreater(abs(float(actual[4]-altered[4])), 1e-3)
            np.testing.assert_array_equal(graph_action(jnp.full((3, 2), 5.), sub), 0.)

    def test_physics_loss_gradient_matches_learned_bias_difference(self):
        with jax.enable_x64(True):
            c, net, times, payload, sub, mask = self.fixture()
            fn = make_loss_vector(c, 3, 2, 1)
            direction = jnp.array([0.7, -0.3])

            def objective(a):
                model = eqx.tree_at(lambda n: n.output.bias, net, net.output.bias+a*direction)
                return fn(model, times, payload, sub, mask).sum()

            actual = jax.grad(objective)(jnp.array(0.))
            step = 1e-5
            finite = (objective(step)-objective(-step))/(2*step)
            np.testing.assert_allclose(actual, finite, rtol=2e-6, atol=1e-7)


class HeterogeneousReferenceLoaderTests(unittest.TestCase):
    def test_explicit_backend_distinguishes_local_from_matching_native_graph(self):
        c = small_config()
        _, _, points = problem.grid(c)
        initial = problem.initial_fields(c, points)
        # Synthetic states exercise provenance/schema, not solver accuracy.
        fields = np.broadcast_to(initial, (2, 2, *initial.shape)).copy()
        graph = (np.array([0, 1]), np.array([1, 0]), np.array([2., 2.]))
        payload = dict(fields=fields, points=points, time=np.array([0., c.sample.t_max]),
            parameters=np.zeros((2, problem.parameter_count(c))), config_json=c.to_json(),
            graph_row=graph[0], graph_col=graph[1], graph_weight=graph[2])
        native = 'native_nonlocal_fp32_weights_fp64_integration'
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, backend in (('native', native), ('local', 'independent_local_fv')):
                path = root/f'{name}.npz'
                np.savez_compressed(path, **payload, backend=np.asarray(backend))
                self.assertEqual(str(load_reference(path, c, points)['backend']), backend)
            self.assertEqual(str(load_reference(root/'native.npz', c, points, graph)['backend']), native)
            # The local file deliberately has identical graph metadata; this
            # must not suffice to certify a native-reference computation.
            with self.assertRaisesRegex(ValueError, 'requires backend'):
                load_reference(root/'local.npz', c, points, graph)
            np.savez_compressed(root/'missing.npz', **payload)
            with self.assertRaisesRegex(ValueError, 'backend'):
                load_reference(root/'missing.npz', c, points)
            for name, backend in (('unknown', np.asarray('unrecognized_solver')),
                                  ('nonscalar', np.array([native]))):
                path = root/f'{name}.npz'
                np.savez_compressed(path, **payload, backend=backend)
                with self.subTest(backend=name), self.assertRaisesRegex(ValueError, 'backend'):
                    load_reference(path, c, points)


if __name__ == '__main__':
    unittest.main()
