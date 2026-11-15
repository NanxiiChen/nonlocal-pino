"""Temporal ablation: common backbone, independent time jets and complete losses."""
import unittest
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from examples.particle_corrosion2d import network, losses
from examples.particle_corrosion2d.evaluate import equal_point_metrics, validate_monitor
from tests.test_particle_corrosion_losses import fixture


class CeramicTemporalTests(unittest.TestCase):
    def test_explicit_film_retains_legacy_and_joint_common_backbone(self):
        with jax.enable_x64(True):
            c, old, data, times = fixture()
            c.arch.temporal_mode = 'decoder_film'
            film = network.build_network(c, jax.random.key(29))
            for a, b in zip(jax.tree.leaves(old), jax.tree.leaves(film), strict=True):
                if eqx.is_array(a): np.testing.assert_array_equal(a, b)
            np.testing.assert_array_equal(network.predict_fields(old, data, data['graph_points'], times, c),
                                          network.predict_fields(film, data, data['graph_points'], times, c))
            c.arch.temporal_mode = 'joint_query'
            joint = network.build_network(c, jax.random.key(29))
            common = eqx.tree_at(lambda m: m.time, old, None)
            for a, b in zip(jax.tree.leaves(common), jax.tree.leaves(joint.backbone), strict=True):
                if eqx.is_array(a): np.testing.assert_array_equal(a, b)
            self.assertIsNone(joint.backbone.time)

    def test_joint_time_jvp_chunking_and_both_streams(self):
        with jax.enable_x64(True):
            c, _, data, times = fixture(); c.arch.temporal_mode = 'joint_query'
            model = network.build_network(c, jax.random.key(29))
            points = data['graph_points']
            prepared = model.prepare_encoded(network.encode(model, data), points)
            raw, fields, dt = network.time_jets(model, prepared, times, c)
            predict = lambda ts: network.predict_fields(model, data, points, ts, c)
            eps = 1e-5
            np.testing.assert_allclose(dt, (predict(times+eps)-predict(times-eps))/(2*eps),
                                       rtol=3e-6, atol=1e-8)
            split = jnp.concatenate([network.predict_fields(model, data, p[None], times, c)
                                     for p in points], axis=1)
            np.testing.assert_allclose(fields, split, atol=1e-13, rtol=1e-12)
            # Independent single-time calls detect incorrect broadcast/mixing of time axes.
            for i in range(len(times)):
                np.testing.assert_allclose(fields[i], predict(times[i:i+1])[0], atol=1e-13)
            self.assertGreater(float(jnp.linalg.norm(raw[0]-raw[1])), 1e-6)
            for token, weight in (('domain_tokens', 'domain_weights'), ('boundary_tokens', 'boundary_weights')):
                padded = dict(data)
                padded[token] = jnp.concatenate((data[token], jnp.full((1, data[token].shape[1]), 1e4)))
                padded[weight] = jnp.r_[data[weight], 0.]
                np.testing.assert_allclose(fields, network.predict_fields(model, padded, points, times, c),
                                           atol=1e-12, rtol=1e-12)
                altered = dict(data); altered[token] = data[token].at[0, 0].add(.13)
                self.assertGreater(float(jnp.linalg.norm(fields-network.predict_fields(model, altered, points, times, c))), 1e-6)

    def test_joint_five_losses_match_separate_queries_and_gradient(self):
        with jax.enable_x64(True):
            c, _, data, times = fixture(); c.arch.temporal_mode = 'joint_query'
            model = network.build_network(c, jax.random.key(29))
            values = losses.condition_losses(model, times, data, c)
            fields = network.predict_fields(model, data, data['mouth_points'], times, c)
            expected_mouth = [losses.weighted_mean(fields[..., 0]**2, data['mouth_weights']),
                losses.weighted_mean((fields[..., 1]/c.physical.c_le)**2, data['mouth_weights'])]
            initial = network.predict_fields(model, data, data['ic_points'], jnp.zeros(1), c)[0]
            expected_initial = losses.weighted_mean(jnp.sum(((initial-data['initial_ic'])/
                jnp.array([1., c.physical.c_le]))**2, axis=-1), data['ic_weights'])
            np.testing.assert_allclose(values[2:], jnp.array([*expected_mouth, expected_initial]), rtol=1e-12)
            def objective(a):
                changed = eqx.tree_at(lambda m: m.temporal_projection.weight, model,
                                      model.temporal_projection.weight+a*.1)
                return losses.condition_losses(changed, times, data, c).sum()
            eps = 1e-5
            actual = jax.grad(objective)(jnp.asarray(0.))
            np.testing.assert_allclose(actual, (objective(eps)-objective(-eps))/(2*eps), rtol=1e-5, atol=1e-6)
            self.assertGreater(abs(float(actual)), 1e-6)
            gradient = eqx.filter_grad(lambda m: losses.condition_losses(m, times, data, c).sum())(model)
            for stream in gradient.backbone.lifts:
                self.assertGreater(sum(float(jnp.sum(x*x)) for x in jax.tree.leaves(stream)
                                       if eqx.is_array(x)), 0.)

    def test_equal_point_norm_then_channel_mean_not_mean_of_time_errors(self):
        target = np.array([[[1., 2.]], [[10., 20.]]])
        prediction = target+np.array([[[1., 0.]], [[0., 4.]]])
        result = equal_point_metrics(prediction, target)
        expected = np.array([1/np.sqrt(101), 4/np.sqrt(404)])
        np.testing.assert_allclose([result['equal_phi'], result['equal_c']], expected)
        self.assertAlmostEqual(result['channel_mean'], expected.mean())
        with self.assertRaisesRegex(ValueError, 'zero reference'):
            equal_point_metrics(prediction, np.zeros_like(target))

    def test_control_configs_only_change_temporal_mode_and_monitoring(self):
        from examples.particle_corrosion2d.configs.train_time_film import get_config
        from examples.particle_corrosion2d.configs.train_time_joint import get_config as joint
        from examples.particle_corrosion2d.configs.train_default import get_config as default
        a, b, original = get_config(), joint(), default()
        validate_monitor(a, 64)
        with self.assertRaisesRegex(ValueError, 'cohort'):
            validate_monitor(a, 4)
        del a.arch['temporal_mode']; del b.arch['temporal_mode']
        self.assertEqual(a.to_dict(), b.to_dict())
        del a['monitor']; a.log.eval_every = original.log.eval_every
        self.assertEqual(a.to_dict(), original.to_dict())


if __name__ == '__main__': unittest.main()
