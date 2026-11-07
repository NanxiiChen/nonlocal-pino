"""Independent time-polynomial and full physical-balance checks."""
import unittest
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from tests.test_particle_corrosion_losses import fixture
from examples.particle_corrosion2d import network,losses
from examples.particle_corrosion2d.problem import initial_fields
from examples.particle_corrosion2d.time_integral import simpson_queries,simpson_balance


class IntegralTests(unittest.TestCase):
    def test_simpson_is_exact_for_cubic_rates_on_nonuniform_intervals(self):
        with jax.enable_x64(True):
            t=jnp.array([0.,.003,.07,.4,1.]);end,q=simpson_queries(t)
            state=.37-end**4/4
            residual=simpson_balance(state[None,:],q[None,:]**3,jnp.array([.37]),end)
            np.testing.assert_allclose(residual,0.,atol=1e-15,rtol=0.)

    def test_actual_model_integral_matches_independent_kks_matrix_oracle(self):
        with jax.enable_x64(True):
            c,net,data,_=fixture()
            c.geometry.pit_x=.5;c.geometry.pit_radius=.12;c.geometry.interface_k=29.;c.geometry.mouth_half_width=.055
            c.loss.time_form='integral_simpson';c.loss.pde_reduction='mean_square'
            times=jnp.array([.8,0.,.03,.2,1.]);end,queries=simpson_queries(times)
            points=data['graph_points'];cache=net.prepare_encoded(network.encode(net,data),points)
            raw=network.decode(net,cache,queries,c)
            fields=network.physical_fields(raw,c)
            actual=losses.integral_residuals(raw,fields,data,end,c)
            phi=np.asarray(fields[...,0]).T;s=np.asarray(raw[...,1]).T
            p=c.physical;dc=p.c_se-p.c_le;matrix=np.array([[2.,-2.,0.],[0.,-1.3,1.3]]);core=np.array([0,2])
            q=p.c_le*(s-1);pc=phi[core]
            ga=p.kappa*(p.n_ac3*(matrix@phi)+p.n_ac2*2*pc*(1-pc)*(1-2*pc)-p.n_ac1*dc*q[core]*6*pc*(1-pc))
            gc=p.n_ch*(matrix@q)
            initial=initial_fields(c,np.asarray(points));pi=initial[:,0];qi=initial[:,1]-dc*pi*pi*(3-2*pi)-p.c_le
            ga[:,0]=p.kappa*(p.n_ac3*(matrix@pi)+p.n_ac2*2*pi[core]*(1-pi[core])*(1-2*pi[core])-p.n_ac1*dc*qi[core]*6*pi[core]*(1-pi[core]))
            gc[:,0]=p.n_ch*(matrix@qi)
            widths=np.diff(np.asarray(end));expected=[]
            for channel,g in enumerate([ga,gc]):
                increments=(g[:,0:-2:2]+4*g[:,1:-1:2]+g[:,2::2])*widths/6
                expected.append(np.asarray(fields)[2::2,core,channel].T-initial[core,channel,None]+np.cumsum(increments,axis=1))
            for a,b in zip(actual,expected,strict=True):np.testing.assert_allclose(a,b,rtol=1e-12,atol=1e-12)
            vector=losses.condition_losses(net,times,data,c)
            w=np.asarray(data['pde_weights']);w=w/w.sum()
            for i,scale in enumerate([c.loss.ac_scale,c.loss.ch_scale]):
                np.testing.assert_allclose(vector[i],scale*np.mean(w@(expected[i]**2)),rtol=1e-12,atol=1e-12)
            def objective(a):
                changed=eqx.tree_at(lambda n:n.output.bias,net,net.output.bias+jnp.array([a,0.]))
                return losses.condition_losses(changed,times,data,c).sum()
            derivative=jax.grad(objective)(jnp.array(0.));eps=1e-5
            finite=(objective(eps)-objective(-eps))/(2*eps)
            self.assertGreater(abs(float(derivative)),1e-6)
            np.testing.assert_allclose(derivative,finite,rtol=2e-5,atol=1e-7)


if __name__=='__main__':unittest.main()
