"""Weighted objective preserves full CSR, uniform reduction and detached context."""
import unittest
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from tests._legacy.fracture2d import test_fracture2d_low_order_damage as fixture_module
from examples.fracture2d._legacy.low_order_damage import make_loss_fn
from examples.fracture2d._legacy.weighted_damage_l1 import frozen_weights, make_weighted_loss


class WeightedDamageTest(unittest.TestCase):
    def setUp(self):
        self.fixture = fixture_module.LowOrderDamageTest(); self.fixture.setUp()

    def test_weight_normalization_and_explicit_cap(self):
        d = np.array([0., .5, 1.], dtype=np.float64)
        w, info = frozen_weights(d, kind='reciprocal', epsilon=.001, cap=None)
        np.testing.assert_allclose(w, (1/(1-d+.001))/np.mean(1/(1-d+.001)))
        self.assertAlmostEqual(w.mean(), 1.)
        w, info = frozen_weights(d, kind='reciprocal', epsilon=.001, cap=50.)
        self.assertEqual(info['raw_max'], 50.)
        with self.assertRaises(ValueError): frozen_weights(d+1., kind='reciprocal', epsilon=.001, cap=None)

    def test_uniform_reduction_and_frozen_weight_network_gradient(self):
        f = self.fixture
        ctx = f.context([0,1,2], [1/3]*3)
        image, loss = make_weighted_loss(gc0=112.5, gc1=.0648, gain=1., population_size=3)
        _, ordinary = make_loss_fn(gc0=112.5, gc1=.0648, gain=1., population_size=3, norm='l1')
        np.testing.assert_allclose(loss(f.net,{**ctx,'weights':jnp.ones(3)},f.load),ordinary(f.net,ctx,f.load),rtol=1e-13)
        w = jnp.array([1.,2.,10.]); context={**ctx,'weights':w}
        def dense(net):
            d=image(net,ctx)
            r=(112.5+2*f.H)*d+.0648*jnp.asarray(f.D.toarray())@d-2*f.H
            return jnp.sum(w*abs(r))/(w.sum()*2*jnp.sqrt(jnp.mean(f.H**2)))
        np.testing.assert_allclose(loss(f.net,context,f.load),dense(f.net),rtol=1e-13)
        a,b=eqx.filter_grad(loss)(f.net,context,f.load),eqx.filter_grad(dense)(f.net)
        for x,y in zip(jax.tree.leaves(a),jax.tree.leaves(b)):np.testing.assert_allclose(x,y,rtol=1e-12,atol=1e-13)
        gradient=jax.grad(lambda weights:loss(f.net,{**ctx,'weights':weights},f.load))(w)
        np.testing.assert_array_equal(gradient,jnp.zeros_like(w))


if __name__ == '__main__': unittest.main()
