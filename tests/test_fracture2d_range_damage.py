"""Zero damage is not equilibrium at positive H; inequality AL algebra."""
import unittest
import jax
import jax.numpy as jnp
import numpy as np

from examples.fracture2d.range_damage_optimization import (
    augmented_term,damage_diagnostics,range_constraints,update_duals,
)


class RangeDamageTest(unittest.TestCase):
    def test_zero_damage_forcing_and_activity_are_separate(self):
        jax.config.update('jax_enable_x64',True)
        D=jnp.asarray([[1.,-1.],[-1.,1.]])
        H=jnp.asarray([.2,1.])
        result=damage_diagnostics(jnp.zeros(2),H,lambda d:D@d,gc0=2.,gc1=.3,
            volumes=jnp.asarray([1.,2.]),thresholds=[.1,.5])
        self.assertEqual(float(result['AT2_absolute_inf']),2.)
        self.assertEqual(float(result['AT2_relative']),1.)
        self.assertEqual(float(result['AT2_forcing_relative']),1.)
        self.assertEqual(float(result['damage_volume_mean']),0.)
        self.assertEqual(float(result['damage_fraction_ge_0.1']),0.)
        # With zero history too, only the unnormalized residual is defined.
        d=h=jnp.zeros(2)
        np.testing.assert_array_equal((2+2*h)*d+.3*(D@d)-2*h,jnp.zeros(2))

    def test_active_bounds_gradient_dual_update_and_detached_coefficients(self):
        jax.config.update('jax_enable_x64',True)
        H=jnp.asarray([1.,2.,3.]);d=jnp.asarray([-.03,.4,1.02])
        lower=jnp.asarray([.2,.1,0.]);upper=jnp.asarray([0.,.1,.3]);penalty=jnp.asarray(10.)
        gc0=2.;scale=2*np.sqrt(np.mean(np.asarray(H)**2))/gc0
        low,high=-np.asarray(d)/scale,(np.asarray(d)-1)/scale
        lo,hi=np.maximum(np.asarray(lower)+10*low,0),np.maximum(np.asarray(upper)+10*high,0)
        expected=np.mean(lo**2-np.asarray(lower)**2+hi**2-np.asarray(upper)**2)/20
        np.testing.assert_allclose(augmented_term(d,H,lower,upper,penalty,gc0=gc0),expected,rtol=1e-14)
        automatic=jax.grad(lambda value:augmented_term(value,H,lower,upper,penalty,gc0=gc0))(d)
        np.testing.assert_allclose(automatic,(-lo+hi)/(len(d)*scale),rtol=1e-14,atol=1e-15)
        direction=jnp.asarray([.3,-.2,.1]);eps=1e-6
        fd=(augmented_term(d+eps*direction,H,lower,upper,penalty,gc0=gc0)
            -augmented_term(d-eps*direction,H,lower,upper,penalty,gc0=gc0))/(2*eps)
        np.testing.assert_allclose(fd,jnp.sum(automatic*direction),rtol=1e-8,atol=1e-12)
        new_lower,new_upper=update_duals(d,H,lower,upper,penalty,gc0=gc0)
        np.testing.assert_allclose(new_lower,lo,rtol=1e-14)
        np.testing.assert_allclose(new_upper,hi,rtol=1e-14)
        for arg in (1,2,3,4):
            gradient=jax.grad(lambda a,b,c,e,f:augmented_term(a,b,c,e,f,gc0=gc0),argnums=arg)(d,H,lower,upper,penalty)
            np.testing.assert_array_equal(gradient,jnp.zeros_like(gradient))
        np.testing.assert_array_equal(jax.grad(lambda x:jnp.sum(update_duals(x,H,lower,upper,penalty,gc0=gc0)[0]))(d),jnp.zeros(3))
        actual_low,actual_high=range_constraints(d,H,gc0=gc0)
        np.testing.assert_allclose(actual_low,low)
        np.testing.assert_allclose(actual_high,high)


if __name__=='__main__':
    unittest.main()
