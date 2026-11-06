"""CPU checks for geometry-conditioned physical outputs and both PDE operators."""
import unittest

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from ml_collections import ConfigDict

from examples.particle_corrosion2d import losses, local_ad, network


def small_config():
    c = ConfigDict()
    c.geometry = ConfigDict(dict(length=1., height=.6))
    c.physical = ConfigDict(dict(c_le=.035, c_se=1., kappa=2., n_ch=8.,
                                 n_ac1=2.4, n_ac2=4., n_ac3=.002))
    c.sample = ConfigDict(dict(t_max=2.3))
    c.arch = ConfigDict(dict(input_dims=(3, 5), extent=(1., .6), width=8, heads=2,
        condition_depth=1, decoder_depth=1, spatial_features=2, spatial_scale=1.2,
        time_features=2, time_max_frequency=3., output_scale=1., norm_eps=1e-5))
    c.train = ConfigDict(dict(compute_dtype='float64', alpha_gradnorm=.9))
    c.loss = ConfigDict(dict(ac_scale=1.2, ch_scale=.9, neumann_scales=(1., 1.),
                            gradnorm_bias=(1.,)*5, gradnorm_weight_cap=1000.))
    return c


def fixture():
    c = small_config()
    model = network.build_network(c, jax.random.key(29))
    data = {
        'domain_tokens': jnp.array([[.2, .1, .3], [.7, .1, .9], [.2, .4, .95], [.8, .4, .98]]),
        'domain_weights': jnp.array([.1, .2, .15, .1]),
        'boundary_tokens': jnp.array([[0., .3, -1., 0., 0.], [1., .3, 1., 0., 0.],
                                      [.5, 0., 0., -1., 1.], [.3, .3, 0., 1., 2.]]),
        'boundary_weights': jnp.array([.6, .6, .1, .4]),
        'graph_points': jnp.array([[.2, .1], [.6, .2], [.8, .4]]),
        'core_indices': jnp.array([0, 2], dtype=jnp.int32),
        'neighbors': jnp.array([[1, 0], [1, 2]], dtype=jnp.int32),
        'coefficients': jnp.array([[2., 0.], [1.3, 0.]]),
        'pde_weights': jnp.array([.8, .2]),
        'ic_points': jnp.array([[.3, .1], [.7, .4]]),
        'initial_ic': jnp.array([[.7, .784], [.9, .972]]),
        'ic_weights': jnp.array([.3, .7]),
        'mouth_points': jnp.array([[.5, 0.], [.53, 0.]]),
        'mouth_weights': jnp.array([.06, .04]),
        'wall_points': jnp.array([[0., .3], [1., .3]]),
        'wall_normals': jnp.array([[-1., 0.], [1., 0.]]),
        'wall_weights': jnp.array([.6, .6]),
    }
    times = jnp.array([.17, .69])
    return c, model, data, times


class ParticleNetworkTests(unittest.TestCase):
    def test_diffusion_forms_liquid_differences_before_chemical_offset(self):
        c = small_config()
        data = dict(core_indices=jnp.array([0, 2], dtype=jnp.int32),
                    neighbors=jnp.array([[1], [1]], dtype=jnp.int32),
                    coefficients=jnp.array([[2.], [1.]], dtype=jnp.float32))
        values = jnp.array([[.125], [.25], [.5]], dtype=jnp.float32)
        np.testing.assert_array_equal(losses.graph_action(values, data),
                                      losses.graph_action(values+64., data))
        raw = jnp.zeros((1, 3, 2), dtype=jnp.float32).at[0, :, 1].set(
            jnp.array([0., 1e-8, 3e-8], dtype=jnp.float32))
        _, ch = losses.residuals(raw, jnp.zeros_like(raw), data, c)
        expected = c.physical.n_ch*c.physical.c_le*np.array([[-2e-8], [2e-8]])
        np.testing.assert_allclose(ch, expected, rtol=2e-7, atol=1e-15)
        self.assertTrue(np.all(np.abs(np.asarray(ch)) > 0))

    def test_physical_time_jvp_shared_queries_and_stream_padding(self):
        with jax.enable_x64(True):
            c, model, data, times = fixture()
            points = data['graph_points']
            prepared = model.prepare_encoded(network.encode(model, data), points)
            _, fields, dt = network.time_jets(model, prepared, times, c)
            predict = lambda t: network.predict_fields(model, data, points, t, c)
            np.testing.assert_allclose(fields, predict(times), rtol=1e-13, atol=1e-13)
            eps = 1e-5
            finite = (predict(times+eps)-predict(times-eps))/(2*eps)
            np.testing.assert_allclose(dt, finite, rtol=2e-6, atol=1e-9)
            # Splitting query coordinates changes neither attention nor outputs.
            split = jnp.concatenate([network.predict_fields(model, data, point[None], times, c)
                                     for point in points], axis=1)
            np.testing.assert_allclose(fields, split, rtol=1e-13, atol=1e-13)
            padded = dict(data)
            for token, measure in (('domain_tokens', 'domain_weights'),
                                   ('boundary_tokens', 'boundary_weights')):
                padded[token] = jnp.concatenate((data[token], jnp.full((1, data[token].shape[1]), 1e4)))
                padded[measure] = jnp.r_[data[measure], 0.]
            np.testing.assert_allclose(fields, network.predict_fields(model, padded, points, times, c),
                                       rtol=1e-12, atol=1e-12)
            altered = dict(data)
            altered['boundary_tokens'] = data['boundary_tokens'].at[-1, 0].add(.13)
            self.assertGreater(float(jnp.linalg.norm(fields-network.predict_fields(model, altered, points, times, c))), 1e-6)

    def test_core_graph_physics_and_five_losses_match_independent_oracle(self):
        with jax.enable_x64(True):
            c, model, data, times = fixture()
            actual = losses.condition_losses(model, times, data, c)
            points = data['graph_points']
            predict = lambda ts: network.predict_fields(model, data, points, ts, c)
            values = np.asarray(predict(times))
            eps = 1e-5
            dt = np.asarray((predict(times+eps)-predict(times-eps))/(2*eps))
            phi, conc = values[..., 0].T, values[..., 1].T
            p = c.physical
            q = conc-(p.c_se-p.c_le)*phi**2*(3-2*phi)-p.c_le
            matrix = np.array([[2., -2., 0.], [0., -1.3, 1.3]])
            core = np.array([0, 2])
            pc, qc = phi[core], q[core]
            ac = dt[:, core, 0].T+p.kappa*(p.n_ac3*(matrix@phi)+p.n_ac2*2*pc*(1-pc)*(1-2*pc)
                                          -p.n_ac1*qc*(p.c_se-p.c_le)*6*pc*(1-pc))
            ch = dt[:, core, 1].T+p.n_ch*(matrix@q)
            initial = np.asarray(network.predict_fields(model, data, data['ic_points'], jnp.zeros(1), c)[0])
            mouth = np.asarray(network.predict_fields(model, data, data['mouth_points'], times, c))
            def average(a, weights):
                return np.mean(np.sum(a*np.asarray(weights), axis=-1)/np.sum(np.asarray(weights)))
            expected = np.array([
                c.loss.ac_scale*average(np.abs(ac).T, data['pde_weights']),
                c.loss.ch_scale*average(np.abs(ch).T, data['pde_weights']),
                average(mouth[..., 0]**2, data['mouth_weights']),
                average((mouth[..., 1]/p.c_le)**2, data['mouth_weights']),
                average(np.sum(((initial-np.asarray(data['initial_ic']))/[1., p.c_le])**2, axis=-1), data['ic_weights'])])
            np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-8)
            np.testing.assert_allclose(losses.graph_action(jnp.asarray(phi), data), matrix@phi, atol=1e-14)
            np.testing.assert_array_equal(losses.graph_action(jnp.ones_like(jnp.asarray(phi)), data), 0.)
            padded = dict(data)
            padded['core_indices'] = jnp.r_[data['core_indices'], 0]
            padded['neighbors'] = jnp.vstack((data['neighbors'], jnp.array([[0, 0]], dtype=jnp.int32)))
            padded['coefficients'] = jnp.vstack((data['coefficients'], jnp.zeros((1, 2))))
            padded['pde_weights'] = jnp.r_[data['pde_weights'], 0.]
            np.testing.assert_allclose(actual, losses.condition_losses(model, times, padded, c), atol=1e-13)
            changed = dict(data)
            changed['initial_ic'] = jnp.zeros_like(data['initial_ic'])
            other = losses.condition_losses(model, times, changed, c)
            np.testing.assert_array_equal(actual[:4], other[:4])
            self.assertGreater(float(jnp.abs(actual[4]-other[4])), .01)
            batch = jax.tree.map(lambda a: jnp.stack((a, a)), data)
            np.testing.assert_allclose(losses.make_loss_vector(c)(model, times, batch), actual, rtol=1e-12, atol=1e-12)

    def test_local_derivatives_match_pointwise_physical_hessian_and_normal(self):
        with jax.enable_x64(True):
            c, model, data, times = fixture()
            encoded = network.encode(model, data)
            points = data['graph_points'][data['core_indices']]

            def point(x):
                raw = local_ad.encoded_raw(model, encoded, x[None], times, c)[:, 0]
                physical = network.physical_fields(raw, c)
                phi = physical[..., 0]
                q = physical[..., 1]-(c.physical.c_se-c.physical.c_le)*phi**2*(3-2*phi)-c.physical.c_le
                return jnp.stack((phi, q), axis=-1)

            hessian = jax.vmap(jax.jacfwd(jax.jacrev(point)))(points)
            expected_lap = jnp.trace(hessian, axis1=-2, axis2=-1).transpose(1, 0, 2)
            actual_lap = local_ad.spatial_laplacians(model, encoded, points, times, c)
            np.testing.assert_allclose(actual_lap, expected_lap, rtol=1e-9, atol=1e-11)
            jacobian = jax.vmap(jax.jacfwd(point))(data['wall_points'])
            expected_normal = jnp.einsum('ntci,ni->tnc', jacobian, data['wall_normals'])
            expected_normal = expected_normal*c.geometry.length/jnp.array([1., c.physical.c_le])
            normal = local_ad.normal_derivatives(model, encoded, data['wall_points'], data['wall_normals'], times, c)
            np.testing.assert_allclose(normal, expected_normal, rtol=1e-9, atol=1e-11)
            local_values = local_ad.condition_losses(model, times, data, c)
            nonlocal_values = losses.condition_losses(model, times, data, c)
            np.testing.assert_allclose(local_values[2:5], nonlocal_values[2:5], rtol=1e-12, atol=1e-12)
            expected_wall = np.mean(np.sum(np.asarray(expected_normal)**2*
                np.asarray(data['wall_weights'])[None, :, None], axis=1)/np.sum(np.asarray(data['wall_weights'])), axis=0)
            np.testing.assert_allclose(local_values[-2:], expected_wall, rtol=1e-10, atol=1e-11)

    def test_parameter_gradient_reaches_geometry_encoder(self):
        with jax.enable_x64(True):
            c, model, data, times = fixture()
            direction = jnp.linspace(-.2, .3, model.lifts[1].layers[-1].bias.size)

            def objective(a):
                changed = eqx.tree_at(lambda net: net.lifts[1].layers[-1].bias, model,
                    model.lifts[1].layers[-1].bias+a*direction)
                return losses.condition_losses(changed, times, data, c).sum()

            actual = jax.grad(objective)(jnp.asarray(0.))
            eps = 1e-5
            finite = (objective(eps)-objective(-eps))/(2*eps)
            self.assertGreater(abs(float(actual)), 1e-5)
            np.testing.assert_allclose(actual, finite, rtol=2e-6, atol=2e-7)


class ParticleGradnormCompilationTests(unittest.TestCase):
    def test_sequential_map_matches_unrolled_complete_adam_update(self):
        """Both refresh branches and both loss counts preserve the full update."""
        for count in (5, 7):
            c = small_config()
            c.loss.gradnorm_bias = tuple(np.linspace(.7, 1.3, count))
            model = eqx.nn.Linear(3, 2, key=jax.random.key(123))
            optimizer = optax.chain(optax.clip_by_global_norm(.8), optax.adam(1e-3))
            state = optimizer.init(eqx.filter(model, eqx.is_array))
            weights = jnp.linspace(.8, 1.2, count, dtype=jnp.float32)
            points = jnp.array([[.1, .3, -.4], [.7, -.2, .5], [-.8, .6, .9]], dtype=jnp.float32)
            targets = jnp.linspace(-.6, .7, count, dtype=jnp.float32)
            projections = jnp.stack((jnp.linspace(.2, .8, count),
                                     jnp.linspace(-.7, .4, count)), axis=-1)
            times = jnp.array([.13, .71], dtype=jnp.float32)

            def vector(net, ts, data):
                predicted = jax.vmap(net)(data)
                return jnp.mean((predicted@projections.T-targets-.1*jnp.mean(ts))**2, axis=0)

            compact = losses.make_step(vector, optimizer, c, tuple(str(i) for i in range(count)))
            for refresh in (True, False):
                values, pullback = eqx.filter_vjp(lambda net: vector(net, times, points), model)
                if refresh:
                    norms = []
                    for i in range(count):
                        gradient = pullback(jax.nn.one_hot(i, count, dtype=values.dtype))[0]
                        norms.append(sum(jnp.sum(jnp.abs(leaf)) for leaf in jax.tree.leaves(gradient)
                                         if eqx.is_array(leaf)))
                    norms = jnp.clip(jnp.stack(norms), 1e-6, 1e6)
                    proposal = jnp.clip(jnp.mean(norms)/norms, 1e-6, c.loss.gradnorm_weight_cap)
                    expected_weights = c.train.alpha_gradnorm*proposal+(1-c.train.alpha_gradnorm)*weights
                else:
                    expected_weights = weights
                gradient = pullback(expected_weights*jnp.asarray(c.loss.gradnorm_bias, dtype=jnp.float32))[0]
                updates, expected_state = optimizer.update(gradient, state, model)
                expected_model = eqx.apply_updates(model, updates)
                actual_model, actual_state, actual_weights, actual_values, finite = compact(
                    model, state, weights, times, points, jnp.asarray(refresh))
                self.assertTrue(bool(finite))
                with self.subTest(count=count, refresh=refresh):
                    np.testing.assert_allclose(actual_values, values, rtol=3e-6, atol=2e-7)
                    np.testing.assert_allclose(actual_weights, expected_weights, rtol=3e-6, atol=2e-7)
                    for actual, expected in zip(jax.tree.leaves((actual_model, actual_state)),
                                                jax.tree.leaves((expected_model, expected_state)), strict=True):
                        np.testing.assert_allclose(actual, expected, rtol=3e-6, atol=2e-7)


if __name__ == '__main__':
    unittest.main()
