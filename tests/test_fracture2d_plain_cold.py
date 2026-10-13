"""Fresh decoder initialization is independent of old weights/physical fields."""
import unittest
from unittest.mock import patch
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from scipy.sparse import csr_matrix

from examples.fracture2d.configs.plain_sigmoid_cold import get_config
from examples.fracture2d.plain_sigmoid_cold import fresh_network,initial_residual_scales
from examples.fracture2d.plain_sigmoid_damage import make_problem,latent_basis_scaling


class PlainColdTest(unittest.TestCase):
    def test_fresh_initial_damage_seed_and_trainable_parameters(self):
        jax.config.update('jax_enable_x64',True)
        c=get_config()
        B=csr_matrix(np.asarray([[1.,.2],[.3,1.],[.5,.4]]))
        net=fresh_network(c,2,B)
        np.testing.assert_array_equal(net.readout.coefficients,jnp.zeros(2))
        np.testing.assert_array_equal(net.trunk.output.weight,jnp.zeros_like(net.trunk.output.weight))
        previous=net
        c.train.seed+=1
        different=fresh_network(c,2,B)
        self.assertFalse(np.array_equal(previous.trunk.layers[0].weight,different.trunk.layers[0].weight))
        net,columns=latent_basis_scaling(net,B,np.ones(2))
        s=dict(matrix=B,operator=None,reference_load=c.cold.reference_load,
               features=lambda H,load:jnp.column_stack((H,H**2)))
        gain=(c.physical.Gc/c.physical.l0)/(c.physical.E*c.cold.reference_load**2)
        with patch('examples.fracture2d.plain_sigmoid_damage.make_padded_matvec',
                   return_value=(lambda d:jnp.asarray([[2.,-1.,-1.],[-1.,2.,-1.],[-1.,-1.,2.]])@d,{})):
            image,_,objectives,_=make_problem(c,s,columns,gain)
        H,load=jnp.asarray([1.,2.,4.]),jnp.asarray(.005)
        np.testing.assert_allclose(image(net,H,load),np.full(3,.01),rtol=1e-13)
        grad=eqx.filter_grad(objectives['residual_l16'])(net,H,load)
        self.assertGreater(float(jnp.linalg.norm(grad.trunk.output.weight)),0.)
        self.assertGreater(float(jnp.linalg.norm(grad.readout.coefficients)),0.)

    def test_initial_jacobian_normalization_matches_independent_algebra(self):
        B=csr_matrix(np.asarray([[1.,.2],[.3,1.],[.5,.4]]))
        D=csr_matrix(np.asarray([[2.,-1.,-1.],[-1.,2.,-1.],[-1.,-1.,2.]]))
        H=np.asarray([1.,2.,4.])
        scales=initial_residual_scales(B,D,H,initial_damage=.01,gc0=2.,gc1=.3)
        A=np.diag(2+2*H)+.3*D.toarray()
        jacobian=A@(.01*.99*B.toarray())*scales
        np.testing.assert_allclose(np.mean((jacobian/(2*np.sqrt(np.mean(H**2))))**2,axis=0),
                                   np.ones(2),rtol=1e-13)


if __name__=='__main__':
    unittest.main()
