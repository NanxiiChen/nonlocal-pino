"""Conservation, boundary geometry, IC compatibility and coupled AD validation."""
import unittest
import numpy as np
from scipy.integrate import quad
import jax
import jax.numpy as jnp
import equinox as eqx
from examples.confined_ch2d.configs.train_base import get_config
from examples.confined_ch2d.problem import initial_field,initial_laplacian,boundary_samples,validate_config
from examples.confined_ch2d.geometry import quadrature,visible_pairs
from examples.confined_ch2d.operators import RegionalLaplacian
from examples.confined_ch2d.native_reference import native_matrix
from examples.confined_ch2d.network import PhaseNetwork,predict_fields
from examples.confined_ch2d.losses import make_loss_fn


def fixture():
    c=get_config();c.nl.delta=.2
    xx,yy=np.meshgrid(np.linspace(.08,.92,12),np.linspace(.17,.83,10))
    x=np.stack((xx.ravel(),yy.ravel()),axis=-1)
    keep=np.sum(((x-np.array(c.geometry.center))/np.array(c.geometry.axes))**2,axis=1)<.95**2
    for cx,cy,r in c.geometry.holes:keep &= np.sum((x-np.array([cx,cy]))**2,axis=1)>(r+.005)**2
    x=x[keep];v=.003*(1+.2*np.sin(13*x[:,0]))
    return c,x,v


class ConfinedCHTests(unittest.TestCase):
    def test_geometric_visibility_quadrature_and_kernel_moment(self):
        c=get_config();validate_config(c)
        pts=np.array([[0.,0.],[2.,0.],[0.,1.]])
        x,v=quadrature(pts,np.array([[0,1,2]]))
        np.testing.assert_allclose(x,[[2/3,1/3]]);np.testing.assert_allclose(v,[1.])
        pts=np.array([[-1.,0.],[1.,0.],[-1.,2.],[1.,2.]])
        np.testing.assert_array_equal(visible_pairs(pts,np.array([[0,1],[2,3],[1,0]]),[(0.,0.,.5)]),[False,True,False])
        d=.07
        moment=np.pi*quad(lambda r:120/(np.pi*d**4)*(1-r/d)**2*r**3,0,d)[0]
        self.assertAlmostEqual(moment,2.,places=13)
        bad=get_config();bad.geometry.holes=((.5,.5,.6),)
        with self.assertRaises(ValueError):validate_config(bad)

    def test_independent_native_matrix_and_weighted_energy_identity(self):
        c,x,v=fixture()
        with jax.enable_x64(True):
            D=RegionalLaplacian(x,v,c);A=native_matrix(x,v,c).toarray()
            actual=np.stack([np.asarray(D@jnp.asarray(e)) for e in np.eye(len(x))],axis=1)
            np.testing.assert_allclose(actual,A,atol=1e-12,rtol=1e-13)
            np.testing.assert_allclose(D@jnp.ones(len(x)),0.,atol=0)
            L=v[:,None]*actual
            np.testing.assert_allclose(L,L.T,atol=1e-14)
            np.testing.assert_allclose(v@actual,0.,atol=1e-13)
            phi=jnp.asarray(.2*np.sin(19*x[:,0])+np.cos(13*x[:,1]));mass=jnp.asarray(v)
            def energy(p):return jnp.sum(mass*(.25*c.physical.beta*(p*p-1)**2+.5*c.physical.epsilon**2*p*(D@p)))
            mu=c.physical.beta*(phi**3-phi)+c.physical.epsilon**2*(D@phi)
            np.testing.assert_allclose(jax.grad(energy)(phi),mass*mu,atol=1e-13,rtol=1e-13)
            rate=-c.physical.mobility*(D@mu)
            self.assertLessEqual(float(jnp.dot(jax.grad(energy)(phi),rate)),0.)
            self.assertLess(abs(float(jnp.dot(mass,rate))),1e-13)

    def test_analytic_initial_laplacian_against_automatic_derivatives(self):
        c,x,_=fixture()
        with jax.enable_x64(True):
            hessian=jax.vmap(jax.hessian(lambda p:initial_field(p,c,xp=jnp)))(jnp.asarray(x))
            expected=np.trace(np.asarray(hessian),axis1=-2,axis2=-1)
        np.testing.assert_allclose(initial_laplacian(x,c),expected,atol=1e-10,rtol=1e-12)

    def test_wall_compatible_initial_field_and_network_mass_jvp(self):
        c,x,v=fixture();b,n,_,_=boundary_samples(c)
        c.arch.hidden_dim=8;c.arch.num_layers=2
        with jax.enable_x64(True):
            f=lambda point:initial_field(point,c,xp=jnp)
            grad=jax.vmap(jax.grad(f))(jnp.asarray(b))
            self.assertLess(float(np.max(abs(np.asarray(grad)*n))),1e-10)
            selected=np.array([0,80,191,192,220,255,256,287,319])
            def mu0(point):
                p=f(point)
                return c.physical.beta*(p**3-p)-c.physical.epsilon**2*jnp.trace(jax.hessian(f)(point))
            dg=jax.vmap(jax.grad(mu0))(jnp.asarray(b[selected]))
            self.assertLess(float(np.max(abs(np.sum(np.asarray(dg)*n[selected],axis=1)))),1e-7)
            net=PhaseNetwork(c,jax.random.PRNGKey(42));points=jnp.asarray(x);vol=jnp.asarray(v)
            mean=jnp.dot(vol,initial_field(points,c,xp=jnp))/vol.sum()
            np.testing.assert_allclose(predict_fields(net,points,0.,mean,vol)[:,0],initial_field(points,c,xp=jnp),atol=1e-14)
            value,dt=jax.jvp(lambda t:predict_fields(net,points,t,mean,vol),(jnp.asarray(.7),),(jnp.asarray(1.),))
            np.testing.assert_allclose(jnp.dot(vol,value[:,0])/vol.sum(),mean,atol=1e-14)
            self.assertLess(abs(float(jnp.dot(vol,dt[:,0]))),1e-14)

    def test_both_residuals_have_finite_coupled_parameter_gradients(self):
        c,x,v=fixture();c.arch.hidden_dim=8;c.arch.num_layers=2
        b,n,bw,_=boundary_samples(c)
        with jax.enable_x64(True):
            net=PhaseNetwork(c,jax.random.PRNGKey(3));points=jnp.asarray(x);vol=jnp.asarray(v)
            mean=jnp.dot(vol,initial_field(points,c,xp=jnp))/vol.sum()
            for method in ('nonlocal','local_ad'):
                c.nl.method=method;D=RegionalLaplacian(x,v,c) if method=='nonlocal' else None
                loss=make_loss_fn(points,vol,D,jnp.asarray(b[::20]),jnp.asarray(n[::20]),jnp.asarray(bw[::20]),c)
                (value,parts),grad=eqx.filter_value_and_grad(loss,has_aux=True)(net,jnp.asarray([.3,.8]),mean)
                self.assertTrue(np.isfinite(float(value)));self.assertEqual(parts.shape,(2 if D is not None else 4,))
                leaves=jax.tree.leaves(eqx.filter(grad,eqx.is_array))
                self.assertTrue(all(np.isfinite(a).all() for a in leaves));self.assertGreater(sum(float(jnp.sum(a*a)) for a in leaves),0.)


if __name__=='__main__':unittest.main()
