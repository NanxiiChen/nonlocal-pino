"""Independent objective/gradient checks and cross-norm selection safeguards."""
import unittest
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from tests import test_fracture2d_low_order_damage as fixtures
from examples.fracture2d.damage_norm import make_loss_fn, eligible_l16
from examples.fracture2d.compact_damage_features import residual_power_loss


class DamageNormTest(unittest.TestCase):
    def test_dense_L1_L16_values_gradients_and_detached_context(self):
        f=fixtures.LowOrderDamageTest();f.setUp();ctx=f.context([0,1,2],[1/3]*3)
        for norm in ['l1','l16']:
            image,loss=make_loss_fn(gc0=112.5,gc1=.0648,gain=1.,population_size=3,norm=norm,power=16,floor=1e-15)
            def dense(net):
                d=image(net,ctx)
                rho=((112.5+2*f.H)*d+.0648*jnp.asarray(f.D.toarray())@d-2*f.H)/(2*jnp.sqrt(jnp.mean(f.H**2)))
                return jnp.mean(abs(rho)) if norm=='l1' else (jnp.mean(rho**16)+1e-240)**(1/8)
            np.testing.assert_allclose(loss(f.net,ctx,f.load),dense(f.net),rtol=1e-13)
            a,b=eqx.filter_grad(loss)(f.net,ctx,f.load),eqx.filter_grad(dense)(f.net)
            for x,y in zip(jax.tree.leaves(a),jax.tree.leaves(b)):
                np.testing.assert_allclose(x,y,rtol=1e-11,atol=1e-13)
            floats={k:v for k,v in ctx.items() if jnp.issubdtype(v.dtype,jnp.inexact)}
            integers={k:v for k,v in ctx.items() if k not in floats}
            grad=jax.grad(lambda values:loss(f.net,{**integers,**values},f.load))(floats)
            for value in grad.values():np.testing.assert_array_equal(value,jnp.zeros_like(value))
        zero=jnp.zeros(3)
        grad=jax.grad(lambda r:residual_power_loss(r,f.H,power=16,floor=1e-15))(zero)
        self.assertTrue(np.isfinite(grad).all());np.testing.assert_array_equal(grad,zero)

    def test_selection_uses_common_field_and_residual_not_loss(self):
        rows=[dict(case='l1_adam',same_H_damage_L2=.05,AT2_relative=.2,loss=1e-10),
              dict(case='l16_adam',same_H_damage_L2=.04,AT2_relative=.3,loss=1e-20),
              dict(case='l16_lbfgs',same_H_damage_L2=.03,AT2_relative=.1,loss=100.)]
        self.assertEqual(eligible_l16(rows),'l16_lbfgs')
        rows[-1]['same_H_damage_L2']=.06
        self.assertIsNone(eligible_l16(rows))


if __name__=='__main__':unittest.main()
