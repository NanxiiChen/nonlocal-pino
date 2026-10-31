"""Optional Fourier bandwidth keeps legacy parameters and time differentiation."""
import unittest
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from nonlocal_pinn.nn.point_conditional import HeterogeneousPointOperator,PointFieldOperator


class PointBandwidthTests(unittest.TestCase):
    def test_explicit_old_band_is_identical_and_has_no_new_parameter_leaves(self):
        with jax.enable_x64(True):
            common=dict(extent=(1.,1.),width=8,heads=2,decoder_depth=2,
                spatial_features=3,spatial_scale=1.,time_features=4,output_scale=.1,
                norm_eps=1e-5,key=jax.random.key(42))
            for cls,extra in ((PointFieldOperator,dict(input_dim=5,latent_tokens=0,encoder_depth=1)),
                              (HeterogeneousPointOperator,dict(input_dims=(5,3),condition_depth=1))):
                old=cls(**common,**extra)
                explicit=cls(**common,**extra,time_max_frequency=4.)
                old_leaves=[a for a in jax.tree.leaves(old) if eqx.is_array(a)]
                new_leaves=[a for a in jax.tree.leaves(explicit) if eqx.is_array(a)]
                self.assertEqual(len(old_leaves),len(new_leaves))
                for a,b in zip(old_leaves,new_leaves,strict=True):np.testing.assert_array_equal(a,b)
                prepared=jax.random.normal(jax.random.key(4),(3,8))
                times=jnp.array([0.,.01,.17,.5,1.])
                np.testing.assert_array_equal(old.decode(prepared,times),explicit.decode(prepared,times))

    def test_wide_bounded_band_time_jvp_and_invalid_metadata(self):
        with jax.enable_x64(True):
            args=dict(input_dims=(5,3),extent=(1.,1.),width=8,heads=2,condition_depth=1,
                decoder_depth=2,spatial_features=3,spatial_scale=1.,time_features=32,
                time_max_frequency=4.,output_scale=.1,norm_eps=1e-5,key=jax.random.key(42))
            net=HeterogeneousPointOperator(**args)
            self.assertEqual(net.time.layers[0].weight.shape[1],65)
            prepared=jax.random.normal(jax.random.key(4),(3,8))
            at=lambda t:net.decode(prepared,t[None])[0]
            t=jnp.array(.37);step=1e-5
            _,dt=jax.jvp(at,(t,),(jnp.ones_like(t),))
            finite_difference=(at(t+step)-at(t-step))/(2*step)
            np.testing.assert_allclose(dt,finite_difference,rtol=2e-5,atol=1e-7)
            for maximum in (0.,float('inf'),float('nan'),True):
                with self.subTest(maximum=maximum),self.assertRaises(ValueError):
                    HeterogeneousPointOperator(**dict(args,time_max_frequency=maximum))


if __name__=='__main__':unittest.main()
