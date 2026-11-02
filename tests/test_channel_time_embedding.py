"""General early/late temporal features, exact legacy/initial behavior and PDE gradients."""
import unittest
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from examples.channel_nsch2d.configs.train_single_dual_time import get_config
from examples.channel_nsch2d.problem import validate_config
from examples.channel_nsch2d.network import temporal_features,fields,encode
from examples.channel_nsch2d.data import geometry_data
from examples.channel_nsch2d.losses import make_loss_fn
from tests.test_channel_nsch import small


class ChannelTimeEmbedding(unittest.TestCase):
    def test_legacy_basis_and_dual_time_derivative(self):
        with jax.enable_x64(True):
            c,_,net=small();time=jnp.array([0.,.1,1.,6.]);t=time/6
            f=jnp.linspace(1.,2.,net.time_features)
            expected=jnp.concatenate((t[:,None],jnp.sin(jnp.pi*t[:,None]*f),jnp.cos(jnp.pi*t[:,None]*f)),axis=-1)
            np.testing.assert_array_equal(temporal_features(net,time,c),expected)
            c.arch.time_features=4
            # Match the fixture's two-frequency architecture by rebuilding it
            # from explicit config; no learned field or time prior is fitted.
            from examples.channel_nsch2d.network import build_network
            net=build_network(c,jax.random.PRNGKey(42));c.time_embedding=get_config().time_embedding
            value=temporal_features(net,time,c)
            self.assertEqual(value.shape,(4,9));np.testing.assert_array_equal(value[:,0],t)
            derivative=jax.jacfwd(lambda q:temporal_features(net,q[None],c)[0])(jnp.array(0.))
            log_rate=1/(c.time_embedding.scale*np.log1p(6/c.time_embedding.scale))
            expected=np.r_[1/6,np.pi*np.array([1.,2.])/6,np.pi*np.array([1.,2.])*log_rate,np.zeros(4)]
            np.testing.assert_allclose(derivative,expected,atol=1e-13)
            self.assertTrue(np.isfinite(np.asarray(value)).all())
            c.time_embedding.scale=0
            with self.assertRaisesRegex(ValueError,'positive finite scale'):validate_config(c)

    def test_initial_fields_preserved_and_full_new_pde_gradient_finite(self):
        with jax.enable_x64(True):
            c,_,_=small();c.sample.Nx=16;c.sample.Ny=8;c.sample.Nt=2;c.arch.time_features=4
            c.nl.degree=4;c.nl.neighbors=48;c.nl.weight_exponent=12.
            from examples.channel_nsch2d.network import build_network
            net=build_network(c,jax.random.PRNGKey(42))
            raw,_=geometry_data((.12,.08),c);data={k:jnp.asarray(v) for k,v in raw.items()}
            e=encode(net,data);query=data['points'][::10]
            before=fields(net,e,query,jnp.zeros(1),data['amplitudes'],c)
            c.time_embedding=get_config().time_embedding
            np.testing.assert_array_equal(fields(net,e,query,jnp.zeros(1),data['amplitudes'],c),before)
            c.time_discretization=get_config().time_discretization;c.phase_transport=get_config().phase_transport
            raw,_=geometry_data((.12,.08),c);data={k:jnp.asarray(v) for k,v in raw.items()}
            batch=jax.tree.map(lambda a:a[None],data)
            (loss,_),gradient=eqx.filter_jit(eqx.filter_value_and_grad(make_loss_fn(c),has_aux=True))(net,batch,jnp.array([.05,3.]))
            self.assertTrue(np.isfinite(float(loss)))
            leaves=[v for v in jax.tree.leaves(gradient) if eqx.is_array(v)]
            self.assertTrue(all(np.isfinite(np.asarray(v)).all() for v in leaves))
            self.assertGreater(sum(float(jnp.sum(v*v)) for v in leaves),0.)


if __name__=='__main__':unittest.main()
