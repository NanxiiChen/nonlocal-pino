"""Temporal consistency, state coupling and full mixed-PDE BDF2 gradients."""
import unittest
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from examples.channel_nsch2d.configs.train_single_bdf2 import get_config
from examples.channel_nsch2d.temporal import backward_rate,previous_times,sample_intervals,step_count
from examples.channel_nsch2d.problem import validate_config
from examples.channel_nsch2d.network import fields,field_jets,spatial_field_jets,encode
from examples.channel_nsch2d.losses import make_loss_fn
from tests.test_channel_nsch import small


class ChannelTemporal(unittest.TestCase):
    def test_uniform_nodes_sampling_and_configuration(self):
        c=get_config();validate_config(c);self.assertEqual(step_count(c),120)
        ids,t=sample_intervals(jax.random.PRNGKey(3),c)
        self.assertEqual(ids.dtype,jnp.int32)
        self.assertEqual(t.shape,(4,))
        self.assertTrue(np.all((np.asarray(ids)>=np.arange(4)*30)&(np.asarray(ids)<(np.arange(4)+1)*30)))
        nodes=jnp.arange(1,121,dtype=jnp.float32)*.05
        p,q=previous_times(nodes,.05)
        np.testing.assert_array_equal(p,jnp.arange(120,dtype=jnp.float32)*.05)
        np.testing.assert_array_equal(q,jnp.maximum(jnp.arange(120,dtype=jnp.float32)-1,0)*.05)
        c.time_discretization.dt=.07
        with self.assertRaisesRegex(ValueError,'divide'):validate_config(c)
        c=get_config();c.sample.include_initial_time=True
        with self.assertRaisesRegex(ValueError,'positive intervals'):validate_config(c)
        with self.assertRaises(Exception) as caught:previous_times(jnp.array([.073]),.05)
        self.assertIn('uniform-grid',str(caught.exception))

    def test_euler_start_and_second_order_bdf_polynomials(self):
        with jax.enable_x64(True):
            dt=.1;t=jnp.arange(1,7)*dt;p,q=previous_times(t,dt)
            rate=lambda f:backward_rate(f(t),f(p),f(q),t,dt)
            np.testing.assert_allclose(rate(lambda t:3*jnp.ones_like(t)),0.,atol=1e-14)
            np.testing.assert_allclose(rate(lambda t:t),1.,atol=1e-14)
            expected=np.asarray(2*t).copy();expected[0]=dt
            np.testing.assert_allclose(rate(lambda t:t*t),expected,atol=1e-14)

    def test_gradients_link_every_predecessor(self):
        with jax.enable_x64(True):
            f=lambda state:backward_rate(state[0:1],state[1:2],state[2:3],jnp.array([.2]),.1)[0]
            np.testing.assert_allclose(jax.grad(f)(jnp.array([.4,.3,.2])),[15.,-20.,5.],atol=1e-14)
            f=lambda state:backward_rate(state[0:1],state[1:2],state[2:3],jnp.array([.1]),.1)[0]
            np.testing.assert_allclose(jax.grad(f)(jnp.array([.4,.3,.2])),[10.,-10.,0.],atol=1e-14)

    def test_spatial_identity_and_full_pde_finite_gradient(self):
        with jax.enable_x64(True):
            c,data,net=small();c.time_discretization=get_config().time_discretization
            c.sample.Nt=2;c.operator.train_conditions=1;c.operator.condition_batch=1
            t=jnp.array([.05,2.]);e=encode(net,data)
            get=eqx.filter_jit(lambda n:spatial_field_jets(n,encode(n,data),data['points'],t,data['amplitudes'],c))
            value,dx,dy=get(net)
            old=eqx.filter_jit(lambda n:field_jets(n,encode(n,data),data['points'],t,data['amplitudes'],c))
            expected,_,ex,ey=old(net)
            for a,b in ((value,expected),(dx,ex),(dy,ey)):np.testing.assert_allclose(a,b,atol=1e-12)
            initial=fields(net,e,data['points'],jnp.zeros(1),data['amplitudes'],c)
            np.testing.assert_allclose(initial[0,:,0],data['domain'][:,2],atol=0,rtol=0)
            batch=jax.tree.map(lambda a:a[None],data)
            (loss,parts),gradient=eqx.filter_jit(eqx.filter_value_and_grad(make_loss_fn(c),has_aux=True))(net,batch,t)
            self.assertEqual(parts.shape,(9,));self.assertTrue(np.isfinite(float(loss)))
            self.assertTrue(all(np.isfinite(np.asarray(a)).all() for a in jax.tree.leaves(gradient) if eqx.is_array(a)))


if __name__=='__main__':unittest.main()
