"""Known-minimum optimizer check and finite empirical-screen requirements."""
import unittest
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from examples.fracture2d._legacy.full_damage_optimization import make_lbfgs, choose_optimizer


class Parameters(eqx.Module):
    x: jax.Array


class FullOptimizerTest(unittest.TestCase):
    def test_cached_optimizer_uses_dynamic_frozen_context_and_warm_parameters(self):
        jax.config.update('jax_enable_x64', True)
        net = Parameters(jnp.array([3., -4.]))
        def objective(net, context, load):
            return jnp.sum((net.x-jax.lax.stop_gradient(context))**2)*load
        controls = dict(maxiter=128, maxfun=8192, memory=20, max_linesearch=50, gtol=1e-12, ftol=0.)
        fit, _ = make_lbfgs(net, objective, controls)
        for target in [jnp.array([1., 2.]), jnp.array([-2., 1.])]:
            net, report = fit(net, target, jnp.asarray(1.))
            np.testing.assert_allclose(net.x, target, rtol=0, atol=1e-10)
            self.assertLess(report['loss'], 1e-20)
            self.assertTrue(report['optimizer_success'])

    def test_screen_records_stalled_candidate_without_silent_fallback(self):
        rows = [dict(optimizer='adam', full_loss=.2, optimizer_success=True),
                dict(optimizer='lbfgs', full_loss=.1, optimizer_success=False, message='budget')]
        self.assertEqual(choose_optimizer(rows), 'lbfgs')
        rows[1]['full_loss'] = np.nan
        with self.assertRaises(ValueError): choose_optimizer(rows)


if __name__ == '__main__': unittest.main()
