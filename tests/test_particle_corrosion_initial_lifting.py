"""Initial-map identity and full time-chain-rule checks, independent of solutions."""
import unittest
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from ml_collections import ConfigDict
from tests.test_particle_corrosion_losses import fixture
from examples.particle_corrosion2d import network, losses
from examples.particle_corrosion2d.problem import initial_fields


class InitialLiftingTests(unittest.TestCase):
    def test_hard_four_losses_and_squared_pde_match_physical_oracle(self):
        with jax.enable_x64(True):
            c, model, data, _ = fixture()
            c.geometry.pit_x = .5
            c.geometry.pit_radius = .12
            c.geometry.interface_k = 29.
            c.geometry.mouth_half_width = .055
            c.phase = ConfigDict(dict(initial_lifting=dict(kind='bounded_exponential', time_scale=.01)))
            c.loss.gradnorm_bias = (1.,)*4
            c.loss.pde_reduction = 'mean_square'
            times = jnp.array([.003, .02])
            actual = losses.condition_losses(model, times, data, c)
            self.assertEqual(actual.shape, (4,))
            self.assertEqual(len(losses.loss_names(c)), 4)
            changed = dict(data, initial_ic=jnp.full_like(data['initial_ic'], 100.))
            np.testing.assert_array_equal(actual, losses.condition_losses(model, times, changed, c))
            points = data['graph_points']
            predict = lambda ts: network.predict_fields(model, data, points, ts, c)
            fields = np.asarray(predict(times)); eps = 1e-7
            dt = np.asarray((predict(times+eps)-predict(times-eps))/(2*eps))
            phi, conc = fields[..., 0].T, fields[..., 1].T
            p=c.physical; dc=p.c_se-p.c_le
            q=conc-dc*phi**2*(3-2*phi)-p.c_le
            matrix=np.array([[2.,-2.,0.],[0.,-1.3,1.3]]); core=np.array([0,2])
            pc=phi[core]
            ac=dt[:,core,0].T+p.kappa*(p.n_ac3*(matrix@phi)+p.n_ac2*2*pc*(1-pc)*(1-2*pc)-p.n_ac1*q[core]*dc*6*pc*(1-pc))
            ch=dt[:,core,1].T+p.n_ch*(matrix@q)
            weights=np.asarray(data['pde_weights'])
            weights=weights/weights.sum()
            expected=np.array([c.loss.ac_scale*np.mean(weights@(ac**2)),c.loss.ch_scale*np.mean(weights@(ch**2))])
            np.testing.assert_allclose(actual[:2],expected,rtol=3e-6,atol=1e-7)

    def test_float32_mouth_endpoint_classification(self):
        from examples.particle_corrosion2d.configs.train_physical_hard import get_config
        c = get_config()
        g = c.geometry
        points = jnp.asarray([[g.pit_x-g.mouth_half_width, 0.],
                              [g.pit_x+g.mouth_half_width, 0.]], dtype=jnp.float32)
        np.testing.assert_array_equal(network.initial_phase(points, c), 0.)

    def test_initial_identity_boundedness_and_physical_time_jvp(self):
        with jax.enable_x64(True):
            c, model, data, _ = fixture()
            c.geometry.pit_x = .5
            c.geometry.pit_radius = .12
            c.geometry.interface_k = 29.
            c.geometry.mouth_half_width = .055
            c.phase = ConfigDict(dict(initial_lifting=dict(kind='bounded_exponential', time_scale=.01)))
            c.arch.time_coordinate = ConfigDict(dict(kind='logarithmic', time_scale=.01))
            points = jnp.array([[.5, 0.], [.5, .08], [.5, .12], [.8, .3]])
            expected = initial_fields(c, np.asarray(points))
            predict = lambda ts: network.predict_fields(model, data, points, ts, c)
            actual = predict(jnp.zeros(1))[0]
            np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-14)
            times = jnp.array([0., .003, .02, .3, 1.])
            cached = model.prepare_encoded(network.encode(model, data), points)
            raw, fields, dt = network.time_jets(model, cached, times, c, points)
            eps = 1e-7
            independent = (predict(times+eps)-predict(times-eps))/(2*eps)
            np.testing.assert_allclose(dt, independent, rtol=2e-7, atol=2e-8)
            np.testing.assert_allclose(fields, predict(times), rtol=1e-14, atol=1e-14)
            phi = np.asarray(fields[..., 0]); self.assertTrue(np.all((phi>=0)&(phi<=1)))
            # Effective liquid channel, not the unlifted neural output, enters KKS.
            h = phi**2*(3-2*phi)
            np.testing.assert_allclose(fields[..., 1], c.physical.c_le*raw[..., 1]+(c.physical.c_se-c.physical.c_le)*h)
            # Initial values do not depend on any trained parameter.
            changed = eqx.tree_at(lambda net: net.output.bias, model, model.output.bias+jnp.array([.7, -.4]))
            changed_initial = network.predict_fields(changed, data, points, jnp.zeros(1), c)[0]
            np.testing.assert_allclose(changed_initial, expected, rtol=1e-14, atol=1e-14)
            self.assertGreater(float(jnp.linalg.norm(network.predict_fields(changed, data, points, times, c)-fields)), .01)

    def test_concentration_metric_changes_only_declared_data_losses(self):
        with jax.enable_x64(True):
            c, model, data, times = fixture()
            old = losses.condition_losses(model, times, data, c)
            c.loss.concentration_scale = 1/c.physical.c_le
            np.testing.assert_allclose(losses.condition_losses(model, times, data, c), old, rtol=1e-14, atol=1e-14)
            c.loss.concentration_scale = 1.
            physical = losses.condition_losses(model, times, data, c)
            np.testing.assert_array_equal(physical[:3], old[:3])
            np.testing.assert_allclose(physical[3], old[3]*c.physical.c_le**2, rtol=1e-13)
            self.assertLess(float(physical[4]), float(old[4]))


if __name__ == '__main__':
    unittest.main()
