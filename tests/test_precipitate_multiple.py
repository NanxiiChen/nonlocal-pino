"""Multi-particle IC compatibility, periodicity and coupled gradients."""
import unittest
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from examples.precipitate2d.configs.train_multiple import get_config
from examples.precipitate2d.configs.train_base import get_config as base_config
from examples.precipitate2d.problem import grid, initial_field, validate_config
from examples.precipitate2d.network import PhaseNetwork, predict_fields
from examples.precipitate2d.mechanics import ElasticEquilibrium
from examples.precipitate2d.losses import make_loss_fn
from examples.twophase2d.operators import PeriodicLaplacian


class MultiplePrecipitateTests(unittest.TestCase):
    def test_union_numpy_exact_ic_periodic_mass_and_parameter_gradient(self):
        c = get_config(); c.sample.Nx = c.sample.Ny = 15; c.nl.delta = .15
        c.arch.hidden_dim = 8; c.arch.num_layers = 2
        validate_config(c)
        points = grid(15,1.)
        # Independent scalar loop/union, rather than reusing the training helper.
        truth = []
        for x in points:
            outside = 1.
            for center,level in zip(c.initial.centers,c.initial.levels,strict=True):
                s = sum(np.sin(np.pi*(x[d]-center[d]))**2 for d in range(2))
                outside *= .5*(1+np.tanh((s-level)/c.initial.width))
            truth.append(c.initial.background+(1-c.initial.background)*(1-outside))
        np.testing.assert_allclose(initial_field(points,c,xp=np),truth,atol=5e-16)
        self.assertGreaterEqual(min(truth),c.initial.background)
        self.assertLessEqual(max(truth),1.)
        with jax.enable_x64(True):
            x = jnp.asarray(points); p0 = initial_field(x,c,xp=jnp); mean = p0.mean()
            net = PhaseNetwork(c,jax.random.PRNGKey(42))
            np.testing.assert_allclose(predict_fields(net,x,0.,mean)[:,0],p0,atol=1e-14)
            for offset in ([1.,0.],[0.,1.]):
                np.testing.assert_allclose(jax.vmap(net,in_axes=(0,None))(x,.7),
                    jax.vmap(net,in_axes=(0,None))(x+jnp.asarray(offset),.7),atol=1e-13)
            value,dt = jax.jvp(lambda t:predict_fields(net,x,t,mean),(jnp.asarray(.7),),(jnp.asarray(1.),))
            np.testing.assert_allclose(value[:,0].mean(),mean,atol=1e-15)
            self.assertLess(abs(float(dt[:,0].mean())),1e-15)
            D = PeriodicLaplacian(15,1.,.15,dtype=np.float64)
            loss = make_loss_fn(x,D,ElasticEquilibrium(c),c)
            (total,parts),grad = eqx.filter_value_and_grad(loss,has_aux=True)(net,jnp.asarray([.2,2.]),mean)
            self.assertTrue(np.isfinite(float(total)))
            self.assertEqual(parts.shape,(2,))
            leaves = jax.tree.leaves(eqx.filter(grad,eqx.is_array))
            self.assertTrue(all(np.isfinite(a).all() for a in leaves))
            self.assertGreater(sum(float(jnp.sum(a*a)) for a in leaves),0.)

    def test_legacy_initial_arithmetic_and_invalid_particle_data(self):
        c = base_config(); x = grid(15,1.)
        s = np.sum(np.sin(np.pi*(x-np.asarray(c.initial.center))/c.geometry.length)**2,axis=-1)
        expected = c.initial.background+(1-c.initial.background)*.5*(1-np.tanh((s-c.initial.level)/c.initial.width))
        np.testing.assert_array_equal(initial_field(x,c,xp=np),expected)
        for centers,levels in [(((.2,.3),),(.1,)),
            (((.2,.3),(.2,.3)),(.1,.2)),
            (((.2,.3),(1.2,.3)),(.1,.2)),
            (((.2,.3),(.5,.6)),(.1,)),
            (((.2,.3),(.5,.6)),(.1,-.2))]:
            bad = get_config(); bad.initial.centers = centers; bad.initial.levels = levels
            with self.assertRaises(ValueError):validate_config(bad)


if __name__=='__main__':unittest.main()
