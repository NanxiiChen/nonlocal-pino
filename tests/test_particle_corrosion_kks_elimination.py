"""Independent physical oracle and gradient checks for the bounded row operation."""
import unittest

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from examples.particle_corrosion2d import losses
from tests.test_particle_corrosion_losses import fixture,small_config


class KKSEliminationTests(unittest.TestCase):
    def test_numpy_physical_oracle_and_phase_rate_decoupling(self):
        with jax.enable_x64(True):
            c=small_config();p=c.physical;dc=p.c_se-p.c_le
            raw=np.array([[[-.3,.2],[.1,.5],[.4,.8]],[[.2,.4],[-.2,.3],[.7,.9]]])
            rt=np.array([[[.4,-.7],[.2,.3],[-.6,.8]],[[-.2,.3],[.1,-.4],[.5,-.6]]])
            data=dict(core_indices=jnp.array([0,2]),neighbors=jnp.array([[1,0],[1,2]]),
                      coefficients=jnp.array([[2.,0.],[1.3,0.]]))
            D=np.array([[2.,-2.,0.],[0.,-1.3,1.3]]);ids=np.array([0,2])
            phi=.5*(1+np.tanh(raw[...,0]));hp=6*phi*(1-phi);h=phi**2*(3-2*phi)
            def mapped(rate):
                pt=2*phi*(1-phi)*rate[...,0]
                return np.stack((phi,p.c_le*raw[...,1]+dc*h),axis=-1),np.stack((pt,p.c_le*rate[...,1]+dc*hp*pt),axis=-1)
            fields,ft=mapped(rt)
            ga=p.kappa*(p.n_ac3*(D@phi.T)+p.n_ac2*(2*phi[:,ids]*(1-phi[:,ids])*(1-2*phi[:,ids])).T
                        -p.n_ac1*dc*p.c_le*(raw[:,ids,1]-1).T*hp[:,ids].T)
            gc=p.n_ch*p.c_le*(D@raw[...,1].T)
            rp=ft[:,ids,0].T+ga;rc=ft[:,ids,1].T+gc
            actual=losses.eliminated_residuals(jnp.asarray(raw),jnp.asarray(rt),jnp.asarray(fields),jnp.asarray(ft),data,c)
            np.testing.assert_allclose(actual[0],rp,rtol=2e-13,atol=2e-13)
            np.testing.assert_allclose(actual[1],rc-dc*hp[:,ids].T*rp,rtol=2e-13,atol=2e-13)
            changed=rt.copy();changed[...,0]+=37;_,changed_ft=mapped(changed)
            altered=losses.eliminated_residuals(jnp.asarray(raw),jnp.asarray(changed),jnp.asarray(fields),jnp.asarray(changed_ft),data,c)
            np.testing.assert_array_equal(altered[1],actual[1])
            self.assertGreater(float(np.max(np.abs(np.asarray(altered[0])-np.asarray(actual[0])))),1)
            # Construct rates satisfying both original equations, not a second formula for the new loss.
            exact=rt.copy();exact[:,ids,0]=(-ga/(2*phi[:,ids].T*(1-phi[:,ids].T))).T
            exact[:,ids,1]=((dc*hp[:,ids].T*ga-gc)/p.c_le).T
            _,exact_ft=mapped(exact)
            zero=losses.eliminated_residuals(jnp.asarray(raw),jnp.asarray(exact),jnp.asarray(fields),jnp.asarray(exact_ft),data,c)
            for residual in zero:np.testing.assert_allclose(residual,0,atol=2e-13)

    def test_full_network_loss_changes_only_transport_and_keeps_coefficient_gradient(self):
        with jax.enable_x64(True):
            c,net,data,times=fixture();c.loss.pde_reduction='mean_square'
            original=losses.condition_losses(net,times,data,c)
            c.loss.transport_residual='ac_eliminated'
            actual=losses.condition_losses(net,times,data,c)
            self.assertEqual(losses.loss_names(c)[1],'ch_ac_eliminated')
            np.testing.assert_allclose(actual[0],original[0],rtol=1e-13,atol=1e-13)
            np.testing.assert_allclose(actual[2:],original[2:],rtol=1e-13,atol=1e-13)
            direction=jnp.array([.2,-.1])
            def objective(epsilon):
                altered=eqx.tree_at(lambda m:m.output.bias,net,net.output.bias+epsilon*direction)
                return losses.condition_losses(altered,times,data,c)[1]
            automatic=jax.grad(objective)(0.)
            epsilon=1e-5;finite=(objective(epsilon)-objective(-epsilon))/(2*epsilon)
            np.testing.assert_allclose(automatic,finite,rtol=2e-5,atol=1e-9)


if __name__=='__main__':unittest.main()
