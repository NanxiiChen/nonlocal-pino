"""Exact output-map calculus, retained CH diffusion, and optional PDE integration."""
import unittest
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from examples.channel_nsch2d.configs.train_stream_tanh import get_config as mapped
from examples.channel_nsch2d.configs.train_stream_tanh_preconditioned import get_config as preconditioned
from examples.channel_nsch2d.phase_coordinate import mapped_phase,map_jacobian,transformed_transport,jacobian_diagnostics
from examples.channel_nsch2d.problem import validate_config,initial_field
from examples.channel_nsch2d.network import lift
from examples.channel_nsch2d.losses import make_loss_fn
from tests.test_channel_nsch import small


class PhaseCoordinate(unittest.TestCase):
    def test_host_diagnostics_do_not_overflow_or_hide_underflow(self):
        import json
        actual=jacobian_diagnostics(np.array([1e-42,.4],dtype=np.float32),1.25)
        self.assertGreater(actual['maximum_inverse_factor'],1e40)
        json.dumps(actual,allow_nan=False)
        actual=jacobian_diagnostics(np.array([0.,.4],dtype=np.float32),1.25)
        self.assertIsNone(actual['maximum_inverse_factor'])
        self.assertEqual(actual['inverse_factor_status'],'jacobian_underflow_in_compute_dtype')
        json.dumps(actual,allow_nan=False)
        with self.assertRaises(FloatingPointError):jacobian_diagnostics(np.array([np.nan]),1.25)

    def test_exact_initial_and_inlet_with_finite_latent(self):
        with jax.enable_x64(True):
            c=mapped();validate_config(c)
            x=jnp.array([[.8,.5],[.8,.72],[0.,.4],[2.,.5]])
            t=jnp.array([0.,.4,6.]);raw=jnp.ones((3,4,4))*.7
            value=lift(raw,x,t,jnp.array([.12,.08]),c)
            expected=initial_field(x,c,xp=jnp)
            np.testing.assert_array_equal(value[0,:,0],expected)
            np.testing.assert_array_equal(value[:,2,0],jnp.ones(3))
            self.assertTrue(np.all(np.abs(np.asarray(value[...,0]))<1.25))
            phase,s=mapped_phase(expected,jnp.zeros(4),c.phase_coordinate)
            np.testing.assert_array_equal(phase,expected)
            self.assertTrue(np.isfinite(np.asarray(s)).all())
            c.phase_coordinate.amplitude=1.
            with self.assertRaisesRegex(ValueError,'amplitude'):validate_config(c)

    def test_chain_rule_and_nonfactored_chemical_diffusion(self):
        with jax.enable_x64(True):
            c=preconditioned();validate_config(c);a=c.phase_coordinate.amplitude
            initial=jnp.array([-1.,-.2,1.]);s=jnp.array([-.6,.3,1.1])
            j=map_jacobian(s,a)
            direct=jax.vmap(jax.grad(lambda z:a*jnp.tanh(z)))(s)
            np.testing.assert_allclose(j,direct,atol=3e-16)
            q=jnp.array([.4,-.3,.2]);diffusion=jnp.array([.07,-.04,.03])
            physical=j*q-diffusion
            actual=transformed_transport(physical,j,a,c.residual_transform.minimum_jacobian)
            np.testing.assert_allclose(actual,a*q-a*diffusion/j,atol=3e-16)
            # A frozen Jacobian would leave a spurious derivative even when
            # physical residual is J*q and its exact quotient is constant a*q.
            fn=lambda z:jnp.sum(transformed_transport(map_jacobian(z,a)*q,
                map_jacobian(z,a),a,c.residual_transform.minimum_jacobian))
            np.testing.assert_allclose(jax.grad(fn)(s),0.,atol=1e-15)
            fn=lambda correction:mapped_phase(initial,correction,c.phase_coordinate)[0]
            _,dt=jax.jvp(fn,(s,),(q,))
            latent=jnp.arctanh(initial/a)+s
            np.testing.assert_allclose(dt,map_jacobian(latent,a)*q,atol=3e-16)

    def test_stable_jacobian_and_explicit_saturation_failure(self):
        latent=jnp.array([-10.,10.],dtype=jnp.float32)
        self.assertTrue(np.all(np.asarray(1-jnp.tanh(latent)**2)==0))
        self.assertTrue(np.all(np.asarray(map_jacobian(latent,1.25))>0))
        fn=eqx.filter_jit(lambda j:transformed_transport(jnp.ones_like(j),j,1.25,1e-6))
        np.testing.assert_allclose(fn(jnp.array([.5])),[2.5])
        with self.assertRaises(Exception) as caught:
            fn(jnp.array([1e-9])).block_until_ready()
        self.assertIn('Jacobian',str(caught.exception))

    def test_small_mixed_pde_gradients_and_unaffected_rows(self):
        with jax.enable_x64(True):
            c,data,net=small();c.phase_coordinate=mapped().phase_coordinate
            batch=jax.tree.map(lambda a:a[None],data);t=jnp.array([.3,1.])
            fn=eqx.filter_jit(eqx.filter_value_and_grad(make_loss_fn(c),has_aux=True))
            (base,parts),grad=fn(net,batch,t)
            self.assertTrue(np.isfinite(float(base)))
            self.assertTrue(all(np.isfinite(np.asarray(a)).all() for a in jax.tree.leaves(grad) if eqx.is_array(a)))
            c.residual_transform=preconditioned().residual_transform
            fn=eqx.filter_jit(eqx.filter_value_and_grad(make_loss_fn(c),has_aux=True))
            (total,changed),grad=fn(net,batch,t)
            self.assertTrue(np.isfinite(float(total)))
            np.testing.assert_allclose(changed[1:],parts[1:],atol=1e-10,rtol=1e-12)
            self.assertNotAlmostEqual(float(changed[0]),float(parts[0]),places=7)
            self.assertTrue(all(np.isfinite(np.asarray(a)).all() for a in jax.tree.leaves(grad) if eqx.is_array(a)))


if __name__=='__main__':unittest.main()
