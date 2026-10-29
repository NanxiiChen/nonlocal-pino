"""Sharper multimode physical IC: independent derivatives, compatibility and gradients."""
import unittest
import numpy as np
import jax,jax.numpy as jnp,equinox as eqx
from examples.confined_ch2d.configs.train_coarsening import get_config
from examples.confined_ch2d.problem import initial_field,initial_laplacian,boundary_samples,validate_config
from examples.confined_ch2d.network import PhaseNetwork,predict_fields
from examples.confined_ch2d.operators import RegionalLaplacian
from examples.confined_ch2d.losses import make_loss_fn
from tests.test_confined_ch import fixture


class MultimodeTests(unittest.TestCase):
    def test_physical_ic_derivatives_mass_and_wall_compatibility(self):
        c=get_config();validate_config(c);_,x,v=fixture()
        truth=[]
        for point in x:
            z=(point-np.asarray(c.geometry.center))/np.asarray(c.geometry.axes)
            s=.5*min(c.geometry.axes)*(1-sum(z*z));w=c.initial.wall_width
            E=s**4/(s**4+w**4)
            for cx,cy,r in c.geometry.holes:
                s=(sum((point-np.array([cx,cy]))**2)-r*r)/(2*r)
                E*=s**4/(s**4+w**4)
            P=sum(a*np.cos(np.pi*(kx*z[0]+ky*z[1])+phase) for kx,ky,a,phase in c.initial.waves)
            truth.append(c.initial.amplitude*E*np.tanh(P/c.initial.scale))
        np.testing.assert_allclose(initial_field(x,c,xp=np),truth,atol=5e-15)
        self.assertLessEqual(np.max(np.abs(truth)),c.initial.amplitude)
        c.arch.hidden_dim=8;c.arch.num_layers=2
        with jax.enable_x64(True):
            fun=lambda p:initial_field(p,c,xp=jnp)
            h=jax.vmap(jax.hessian(fun))(jnp.asarray(x))
            np.testing.assert_allclose(initial_laplacian(x,c),np.trace(np.asarray(h),axis1=-2,axis2=-1),atol=1e-8,rtol=1e-11)
            b,n,_,_=boundary_samples(c);b,n=b[::23],n[::23]
            def mu(p):
                phi=fun(p)
                return c.physical.beta*(phi**3-phi)-c.physical.epsilon**2*jnp.trace(jax.hessian(fun)(p))
            for f in (fun,mu):
                g=np.asarray(jax.vmap(jax.grad(f))(jnp.asarray(b)))
                self.assertLess(np.max(np.abs(np.sum(g*n,axis=1))),1e-7)
            net=PhaseNetwork(c,jax.random.PRNGKey(42));points=jnp.asarray(x);vol=jnp.asarray(v)
            mean=jnp.dot(vol,initial_field(points,c,xp=jnp))/vol.sum()
            np.testing.assert_allclose(predict_fields(net,points,0.,mean,vol)[:,0],truth,atol=1e-14)
            value,dt=jax.jvp(lambda t:predict_fields(net,points,t,mean,vol),(jnp.asarray(1.2),),(jnp.asarray(1.),))
            np.testing.assert_allclose(jnp.dot(vol,value[:,0])/vol.sum(),mean,atol=1e-14)
            self.assertLess(abs(float(jnp.dot(vol,dt[:,0]))),1e-14)

    def test_invalid_wave_data_and_both_coupled_gradients(self):
        for waves in [((0,0,1.,0.),(1,2,.4,1.)),((1.5,1,1.,0.),(1,2,.4,1.)),((1,2,1.),(2,3,1.))]:
            c=get_config();c.initial.waves=waves
            with self.assertRaises(ValueError):validate_config(c)
        c=get_config();_,x,v=fixture();c.nl.delta=.2;c.arch.hidden_dim=8;c.arch.num_layers=2
        b,n,bw,_=boundary_samples(c)
        with jax.enable_x64(True):
            net=PhaseNetwork(c,jax.random.PRNGKey(4));points=jnp.asarray(x);vol=jnp.asarray(v)
            mean=jnp.dot(vol,initial_field(points,c,xp=jnp))/vol.sum()
            for method in ('nonlocal','local_ad'):
                c.nl.method=method;D=RegionalLaplacian(x,v,c) if method=='nonlocal' else None
                loss=make_loss_fn(points,vol,D,jnp.asarray(b[::30]),jnp.asarray(n[::30]),jnp.asarray(bw[::30]),c)
                (value,parts),grad=eqx.filter_value_and_grad(loss,has_aux=True)(net,jnp.asarray([.3,1.2]),mean)
                self.assertTrue(np.isfinite(float(value)))
                self.assertEqual(parts.shape,(2 if method=='nonlocal' else 4,))
                leaves=jax.tree.leaves(eqx.filter(grad,eqx.is_array))
                self.assertTrue(all(np.isfinite(a).all() for a in leaves));self.assertGreater(sum(float(jnp.sum(a*a)) for a in leaves),0.)


if __name__=='__main__':unittest.main()
