"""Physical-time chain rule and hard constraints under a shared log coordinate."""
import unittest
import jax
import jax.numpy as jnp
import numpy as np
from examples.confined_ch2d.configs.train_operator_smallscale10_512_logtime import get_config
from examples.confined_ch2d.operator_problem import validate,fixed_initial
from examples.confined_ch2d.operator_training import time_coordinate,phase,fields,query_phase,build_network
from test_confined_ch_operator import small_fixture


class ConstantDecoder:
    def decode(self,prepared,times):return jnp.broadcast_to(prepared,(len(times),len(prepared)))


class OperatorLogTimeTests(unittest.TestCase):
    def test_physical_initial_derivative_and_legacy_expression(self):
        with jax.enable_x64(True):
            c=get_config();validate(c);_,data=small_fixture();raw=jnp.array([1.,-2.,3.,0.])
            centered=raw-jnp.dot(data['volumes'],raw)/data['volumes'].sum()
            at=lambda t:phase(ConstantDecoder(),raw,t,data,c)
            phi,dt=jax.jvp(at,(jnp.array(0.),),(jnp.array(1.),))
            np.testing.assert_array_equal(phi,data['initial'])
            slope=1/(c.phase.initial_time_scale*np.log1p(c.physical.terminal_time/c.phase.initial_time_scale))
            expected=jnp.where(data['volumes']>0,slope*centered,0.)
            np.testing.assert_allclose(dt,expected,atol=1e-14)
            np.testing.assert_array_equal(time_coordinate(jnp.array([0.,3.]),c),jnp.array([0.,1.]))
            for t in (0.,.001,.05,.5,3.):
                np.testing.assert_allclose(at(t)@data['volumes'],data['initial']@data['volumes'],atol=1e-14)
            del c.phase
            times=jnp.array([0.,.001,.05,.5,3.])
            np.testing.assert_array_equal(time_coordinate(times,c),times/c.physical.terminal_time)
            expected=jnp.where(data['volumes']>0,data['initial']+(.5/c.physical.terminal_time)*centered,0.)
            np.testing.assert_array_equal(phase(ConstantDecoder(),raw,.5,data,c),expected)

    def test_real_network_query_consistency_and_time_jvp(self):
        with jax.enable_x64(True):
            c=get_config();c.arch.width=8;c.arch.heads=2;c.arch.condition_depth=1
            c.arch.decoder_depth=1;c.arch.spatial_features=2;c.arch.time_features=2
            _,data=small_fixture();data=dict(data)
            data['initial']=jnp.where(data['volumes']>0,fixed_initial(data['points'],c,xp=jnp),0.)
            net=build_network(c,jax.random.key(42));times=jnp.array([0.,.01,.1,1.,3.])
            output=fields(net,data,times,c)
            np.testing.assert_allclose(query_phase(net,data,data['points'][:3],times,c),output[:,:3,0],atol=1e-13)
            at=lambda t:fields(net,data,t[None],c)[0]
            t=jnp.array(.07);_,dt=jax.jvp(at,(t,),(jnp.array(1.),))
            finite_difference=(at(t+1e-6)-at(t-1e-6))/2e-6
            np.testing.assert_allclose(dt,finite_difference,atol=1e-8,rtol=1e-6)
            c.phase.initial_time_scale=0.
            with self.assertRaises(ValueError):validate(c)


if __name__=='__main__':unittest.main()
