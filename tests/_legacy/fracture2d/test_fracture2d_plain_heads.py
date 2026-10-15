"""Independent head derivatives and matched initial residual sensitivities."""
import unittest
from unittest.mock import patch
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from scipy.sparse import csr_matrix

from examples.fracture2d._legacy.configs.plain_damage_head_trial import get_config
from examples.fracture2d._legacy.plain_damage_head import (
    fresh_network,initial_scales,initialization,make_problem,map_damage,
)


class PlainHeadsTest(unittest.TestCase):
    def test_maps_derivatives_and_saturation(self):
        jax.config.update('jax_enable_x64',True)
        for head in ('sigmoid','exp_square','identity'):
            z,tangent,gain=initialization(head,1e-5)
            self.assertAlmostEqual(float(map_damage(head,jnp.asarray(z))),1e-5,places=16)
            automatic=float(jax.grad(lambda x:map_damage(head,x))(jnp.asarray(z)))
            np.testing.assert_allclose(automatic,tangent,rtol=1e-14)
            np.testing.assert_allclose(gain*automatic,1e-5*(1-1e-5),rtol=1e-14)
        z=-np.sqrt(-np.log(1e-9))
        np.testing.assert_allclose(jax.grad(lambda x:map_damage('exp_square',x))(jnp.asarray(z)),
            -2*z*1e-9,rtol=1e-14)
        self.assertEqual(float(jax.grad(lambda x:map_damage('exp_square',x))(jnp.asarray(0.))),0.)
        self.assertEqual(float(map_damage('identity',jnp.asarray(-.1))),-.1)
        with self.assertRaises(ValueError):
            initialization('invalid',1e-5)
        with self.assertRaises(ValueError):
            initialization('sigmoid',0.)

    def test_initial_fields_jacobians_gradients_and_detached_H_match(self):
        jax.config.update('jax_enable_x64',True)
        c=get_config()
        D=csr_matrix(np.asarray([[2.,-1.,-1.],[-1.,2.,-1.],[-1.,-1.,2.]]))
        B=csr_matrix(np.asarray([[1.,.2],[.3,1.],[.5,.4]]))
        H,load=jnp.asarray([1.,2.,4.]),jnp.asarray(.0001)
        g0,g1=c.physical.Gc/c.physical.l0,c.physical.Gc*c.physical.l0
        reference=None
        for head in c.head.heads:
            columns,gain=initial_scales(B,D,np.asarray(H),head=head,
                initial_damage=c.head.initial_damage,gc0=g0,gc1=g1)
            net=fresh_network(c,2,B,head=head)
            s=dict(matrix=B,operator=None,features=lambda h,l:jnp.column_stack((h,h**2)))
            with patch('examples.fracture2d._legacy.plain_sigmoid_damage.make_padded_matvec',
                       return_value=(lambda d:jnp.asarray(D.toarray())@d,{})),\
                 patch('examples.fracture2d._legacy.plain_damage_head.make_padded_matvec',
                       return_value=(lambda d:jnp.asarray(D.toarray())@d,{})):
                image,metrics,objectives,_=make_problem(c,s,columns,gain,head=head)
            np.testing.assert_allclose(image(net,H,load),np.full(3,1e-5),rtol=1e-13)
            _,tangent,_=initialization(head,1e-5)
            A=np.diag(g0+2*np.asarray(H))+g1*D.toarray()
            normalized=A@(tangent*B.toarray()*columns)/(2*np.sqrt(np.mean(np.asarray(H)**2)))
            np.testing.assert_allclose(np.mean(normalized**2,axis=0),np.ones(2),rtol=1e-13)
            gradient=eqx.filter_grad(objectives['residual_l16'])(net,H,load)
            flat=np.concatenate([np.asarray(a).ravel() for a in jax.tree.leaves(gradient)])
            if reference is None:
                reference=flat
            else:
                np.testing.assert_allclose(flat,reference,rtol=1e-10,atol=1e-12)
            np.testing.assert_array_equal(jax.grad(lambda h:objectives['residual_l16'](net,h,load))(H),jnp.zeros(3))
            eps=1e-6
            plus=eqx.tree_at(lambda n:n.readout.coefficients,net,net.readout.coefficients+jnp.asarray([eps,0.]))
            minus=eqx.tree_at(lambda n:n.readout.coefficients,net,net.readout.coefficients-jnp.asarray([eps,0.]))
            fd=(objectives['residual_l16'](plus,H,load)-objectives['residual_l16'](minus,H,load))/(2*eps)
            np.testing.assert_allclose(gradient.readout.coefficients[0],fd,rtol=1e-5,atol=1e-9)
            self.assertGreater(float(metrics(net,H,load)['damage_min']),0.)


if __name__=='__main__':
    unittest.main()
