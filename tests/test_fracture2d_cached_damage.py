"""Cached compilation preserves solver policy and changing-history dependence."""
import unittest
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from examples.fracture2d.cached_damage_optimization import make_cached_lbfgs
from examples.fracture2d.compact_damage_features import Readout
from examples.fracture2d.damage_optimization import optimize_lbfgs


class CachedDamageTest(unittest.TestCase):
    def test_optimizer_parity_and_dynamic_history(self):
        jax.config.update('jax_enable_x64', True)
        net = Readout(jnp.asarray([.05, .07]))
        def objective(n, H, load):
            return jnp.mean((n.coefficients-(H+load))**2)
        def metrics(n, H, load):
            return dict(AT2_relative=jnp.max(abs(n.coefficients-(H+load))),
                        damage_min=jnp.min(n.coefficients), damage_max=jnp.max(n.coefficients))
        fit, vg, _ = make_cached_lbfgs(net, objective, metrics)
        controls = dict(maxiter=8, check_every=1, memory=20, max_linesearch=30,
                        gtol=1e-12, ftol=0., residual_limit=1e-10, range_tolerance=1e-10)
        for H, load in [(jnp.asarray([.1, .2]), jnp.asarray(.01)),
                        (jnp.asarray([.2, .3]), jnp.asarray(.02))]:
            cached, cr = fit(net, H, load, controls, progress=lambda row: None)
            original, old = optimize_lbfgs(net, objective, metrics, H, load, controls, progress=lambda row: None)
            np.testing.assert_allclose(cached.coefficients, original.coefficients, rtol=1e-13, atol=1e-14)
            for key in ('status', 'iterations', 'evaluations'):
                self.assertEqual(cr[key], old[key])
            np.testing.assert_allclose(cached.coefficients, H+load, atol=1e-12)
            net = cached
        with self.assertRaisesRegex(ValueError, 'structure/shapes/dtypes'):
            fit(Readout(jnp.zeros(3, jnp.float64)), H, load, controls, progress=lambda row: None)
        with self.assertRaisesRegex(ValueError, 'structure/shapes/dtypes'):
            fit(Readout(jnp.zeros(2, jnp.float32)), H, load, controls, progress=lambda row: None)
        arrays, _ = eqx.partition(net, eqx.is_inexact_array)
        from jax.flatten_util import ravel_pytree
        flat, _ = ravel_pytree(arrays)
        value, gradient = vg(flat, H*2, load)
        np.testing.assert_allclose(value, objective(net, H*2, load), atol=1e-15)
        np.testing.assert_allclose(gradient, np.asarray(net.coefficients-(H*2+load)), atol=1e-14)


if __name__ == '__main__':
    unittest.main()
