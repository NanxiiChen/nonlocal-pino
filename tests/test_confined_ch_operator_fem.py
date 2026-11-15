"""Independent P1 weak equations, signed couplings and neural field checks."""
import unittest

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from examples.confined_ch2d.configs.train_operator_compare_fem import get_config
from examples.confined_ch2d.operator_fem_assembly import assemble,build_input
from examples.confined_ch2d.operator_fem_weak import action,make_loss,phase,fields,query_phase_encoded
from examples.confined_ch2d.operator_problem import fixed_initial
from tests.test_confined_ch_operator_local_ad import Polynomial


def fixture():
    c=get_config();c.train.compute_dtype='float64';c.train.seed=42
    c.operator.fem.max_points=8;c.operator.fem.max_neighbors=8;c.operator.online.max_points=4
    vertices=np.array([[0.,0.],[1.,0.],[1.,1.],[0.,1.],[.08,.55]])
    cells=np.array([[0,1,4],[1,2,4],[2,3,4],[3,0,4]])
    mass,K,points,volumes=assemble(vertices,cells)
    payload,report=build_input(vertices,cells,c)
    payload.update(points=points,volumes=volumes,initial=fixed_initial(points,c,xp=np),
        condition=np.zeros((4,5)),condition_weights=np.ones(4))
    return c,vertices,cells,mass,K,{k:jnp.asarray(v) for k,v in payload.items()},report


class ConfinedFEMTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old=jax.config.jax_enable_x64;jax.config.update('jax_enable_x64',True)

    @classmethod
    def tearDownClass(cls):
        jax.config.update('jax_enable_x64',cls.old)

    def test_independent_geometric_assembly_energy_and_resolvent(self):
        c,v,cells,mass,K,data,report=fixture();n=len(v)
        expected=np.zeros((n,n));m=np.zeros(n)
        for ids in cells:
            x=v[ids];det=np.linalg.det(np.stack((x[1]-x[0],x[2]-x[0])))
            grad=np.array([[x[1,1]-x[2,1],x[2,0]-x[1,0]],
                           [x[2,1]-x[0,1],x[0,0]-x[2,0]],
                           [x[0,1]-x[1,1],x[1,0]-x[0,0]]])/det
            area=abs(det)/2
            expected[np.ix_(ids,ids)]+=area*grad@grad.T;m[ids]+=area/3
        np.testing.assert_allclose(K.toarray(),expected,rtol=2e-14,atol=2e-14)
        np.testing.assert_allclose(mass,m,rtol=2e-14)
        self.assertGreater(report['positive_off_diagonals_retained'],0)
        field=np.pad(v@np.array([2.,-3.])+1,(0,3))
        np.testing.assert_allclose(action(jnp.asarray(field),data)[:n],expected@field[:n]/mass,rtol=2e-14,atol=2e-14)
        self.assertAlmostEqual(float(field[:n]@expected@field[:n]),13.,places=12)
        np.testing.assert_allclose(expected@np.ones(n),0,atol=1e-14)
        P=np.asarray(data['fe_metric'])[:n,:n]
        np.testing.assert_allclose(P@np.ones(n),1,atol=1e-12)
        np.testing.assert_allclose(mass[:,None]*P,P.T*mass[None,:],atol=1e-13)
        self.assertGreater(np.linalg.eigvalsh(P*np.sqrt(mass[:,None]/mass[None,:])).min(),0)

    def test_weak_residual_parameter_gradient_and_nodal_mass(self):
        c,vertices,_,mass,K,d,_=fixture();c.loss.initial_residual_fraction=.2
        n=len(mass);D=K.toarray()/mass[:,None]
        times=jnp.array([0.,.2,1.3]);net=Polynomial(jnp.asarray(.12));batch=jax.tree.map(lambda x:x[None],d)
        loss=make_loss(c);actual,gradient=eqx.filter_value_and_grad(loss)(net,batch,times)
        x,y=vertices.T;q=x**4+x*x*y*y+2*y**4;r=x**3*y
        q-=mass@q/mass.sum();r-=mass@r/mass.sum();initial=np.asarray(d['fe_initial'])[:n]
        P=np.asarray(d['fe_metric'])[:n,:n];T=c.physical.terminal_time

        def independent(a):
            values=[]
            for t in np.asarray(times):
                s=t/T;phi=initial+s*(a*q+s*r);dt=(a*q+2*s*r)/T
                mu=c.physical.beta*(phi**3-phi)+c.physical.epsilon**2*(D@phi)
                residual=T*(dt+c.physical.mobility*(D@mu));filtered=P@residual
                values.append(mass@(filtered**2)/mass.sum())
            f=c.loss.initial_residual_fraction
            return f*values[0]+(1-f)*np.mean(values[1:])

        np.testing.assert_allclose(actual,independent(float(net.a)),rtol=3e-12,atol=3e-12)
        h=1e-5;finite=(independent(float(net.a)+h)-independent(float(net.a)-h))/(2*h)
        np.testing.assert_allclose(gradient.a,finite,rtol=3e-8,atol=3e-8)
        c.loss.initial_residual_fraction=0.
        new_value,new_gradient=eqx.filter_value_and_grad(make_loss(c))(net,batch,times)
        np.testing.assert_allclose(new_value,independent(float(net.a)),rtol=3e-12)
        finite=(independent(float(net.a)+h)-independent(float(net.a)-h))/(2*h)
        np.testing.assert_allclose(new_gradient.a,finite,rtol=3e-8,atol=3e-8)
        for t in times:
            predicted=phase(net,d['fe_points'],t,d,c)
            self.assertAlmostEqual(float(d['fe_mass']@(predicted-d['fe_initial'])),0,places=13)

    def test_common_query_fields_and_initial_constraint(self):
        c,_,cells,_,_,d,_=fixture();net=Polynomial(jnp.asarray(.3));times=jnp.array([0.,.4,2.])
        prediction=fields(net,d,times,c)
        np.testing.assert_array_equal(prediction[0,:,0],d['initial'])
        phi=query_phase_encoded(net,None,d,d['points'],times,c)
        np.testing.assert_allclose(prediction[...,0],phi,atol=1e-14,rtol=1e-14)
        for i,t in enumerate(times):
            node=phase(net,d['fe_points'],t,d,c)
            mu=c.physical.beta*(node**3-node)+c.physical.epsilon**2*action(node,d)
            np.testing.assert_allclose(prediction[i,:,1],np.asarray(mu)[cells].mean(axis=1),atol=1e-13)


if __name__=='__main__':unittest.main()
