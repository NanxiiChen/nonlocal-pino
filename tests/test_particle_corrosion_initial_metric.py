"""Equivalent known IC in liquid coordinates, independently checked."""
import unittest
import jax
import jax.numpy as jnp
import numpy as np
from tests.test_particle_corrosion_losses import fixture
from examples.particle_corrosion2d import losses,network


class InitialMetricTests(unittest.TestCase):
    def test_only_ic_changes_and_matches_independent_raw_output_oracle(self):
        with jax.enable_x64(True):
            c,net,data,times=fixture()
            original=losses.condition_losses(net,times,data,c)
            c.loss.initial_metric='phase_liquid'
            actual=losses.condition_losses(net,times,data,c)
            np.testing.assert_array_equal(actual[:4],original[:4])
            prepared=net.prepare_encoded(network.encode(net,data),data['ic_points'])
            raw=np.asarray(network.decode(net,prepared,jnp.zeros(1),c)[0])
            phi=.5*(1+np.tanh(raw[:,0]));target=np.asarray(data['initial_ic'])
            dc=c.physical.c_se-c.physical.c_le
            s0=(target[:,1]-dc*target[:,0]**2*(3-2*target[:,0]))/c.physical.c_le
            weights=np.asarray(data['ic_weights'])
            expected=np.sum(weights*((phi-target[:,0])**2+(raw[:,1]-s0)**2))/weights.sum()
            np.testing.assert_allclose(actual[-1],expected,rtol=1e-13,atol=1e-13)
            target_raw=np.stack((np.arctanh(2*target[:,0]-1),s0),axis=-1)
            np.testing.assert_allclose(network.physical_fields(jnp.asarray(target_raw),c),target,rtol=1e-13,atol=1e-13)

    def test_physical_ic_can_hide_liquid_error_without_changing_zero_set(self):
        cle=5100/143000;dc=1-cle;phi0=.5;c0=.5;s0=.5
        phi=.51;h=lambda p:p*p*(3-2*p)
        s=(c0-dc*h(phi))/cle
        physical=(phi-phi0)**2+((cle*s+dc*h(phi)-c0)/cle)**2
        natural=(phi-phi0)**2+(s-s0)**2
        self.assertGreater(natural,100*physical)
        jac=np.array([[1.,0.],[dc/cle*1.5,1.]])
        self.assertGreater(np.linalg.cond(jac),1000.)
        self.assertAlmostEqual(cle*s0+dc*h(phi0),c0)


if __name__=='__main__':unittest.main()
