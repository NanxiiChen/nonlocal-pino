"""Independent ellipsoid geometry,3Dderivatives/gradients and stencil checks."""
import unittest
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

from nonlocal_pinn.nn.conditional import ConditionalFieldOperator
from nonlocal_pinn.operators import build_D
from examples.icemelting3d.ellipsoids import rotation,signed_distance,signed_distance_numpy
from examples.icemelting3d.configs.train_operator import get_config
from examples.icemelting3d.operator_data import physics,validate_config
from examples.icemelting3d.operator_losses import latent_fields,make_loss_fn
from examples.icemelting3d.operator_reference import coefficients,rhs


class MeltingOperatorTests(unittest.TestCase):
    def test_float32_rotated_distance_near_principal_axes(self):
        p=np.array([.5,.5,.5,.36,.30,.24,.4,.2,.3,.7])
        p[6:]/=np.linalg.norm(p[6:])
        points=np.r_[np.random.default_rng(3).uniform(0,1,(100,3)),
            np.array([[0,0,0],[.1,0,0],[0,.05,0],[0,0,.01]])@rotation(p[6:],np).T+p[:3]]
        # Match the actual float32 geometry metadata/coordinates independently.
        p=p.astype(np.float32);points=points.astype(np.float32)
        expected=signed_distance_numpy(points,p,64)
        actual=jax.jit(lambda x,q:signed_distance(x,q,40))(jnp.asarray(points),jnp.asarray(p))
        np.testing.assert_allclose(actual,expected,rtol=0,atol=3e-7)

    def test_euclidean_sphere_and_ellipsoid_interior_degeneracies(self):
        with jax.enable_x64(True):
            p=np.array([.5,.5,.5,.3,.3,.3,1.,0,0,0])
            points=np.r_[np.random.default_rng(1).uniform(.1,.9,(20,3)),[[.5,.5,.5]]]
            expected=.3-np.linalg.norm(points-.5,axis=1)
            np.testing.assert_allclose(signed_distance_numpy(points,p,56),expected,atol=1e-14)
            np.testing.assert_allclose(signed_distance(jnp.asarray(points),jnp.asarray(p),56),expected,atol=1e-14)
            p[3:6]=[.36,.30,.24]
            points=np.r_[np.random.default_rng(8).uniform(.1,.9,(8,3)),[[.5,.5,.5],[.6,.5,.5],[.5,.55,.5],[.86,.5,.5]]]
            actual=signed_distance_numpy(points,p,56)
            for point,d in zip(points,actual):
                target=point-p[:3];a=p[3:6]
                values=[]
                for start in np.r_[np.diag(a),-np.diag(a)]:
                    result=minimize(lambda x:np.sum((x-target)**2),start,
                        constraints={'type':'eq','fun':lambda x:np.sum((x/a)**2)-1},
                        method='SLSQP',options={'ftol':1e-13,'maxiter':200})
                    self.assertTrue(result.success,result.message)
                    values.append(np.sqrt(result.fun))
                self.assertAlmostEqual(abs(d),min(values),places=7)
            np.testing.assert_allclose(signed_distance(jnp.asarray(points),jnp.asarray(p),56),actual,atol=1e-14)
            quaternion=np.array([.4,.2,.3,.7]);quaternion/=np.linalg.norm(quaternion)
            rotated=p.copy();rotated[6:]=quaternion
            transformed=(points-p[:3])@np.asarray(rotation(quaternion,np)).T+p[:3]
            np.testing.assert_allclose(signed_distance_numpy(transformed,rotated,56),actual,atol=2e-14)

    def test_3d_split_time_jvp_hard_ic_and_parameter_gradient(self):
        with jax.enable_x64(True):
            c=get_config();validate_config(c)
            net=ConditionalFieldOperator(image_shape=(1,4,4,4),patch_shape=(2,2,2),extent=(1.,1.,1.),
                width=12,encoder_depth=1,heads=2,mlp_ratio=2,decoder_depth=2,spatial_features=3,
                spatial_scale=1.,time_features=2,time_scale=1.,local_dim=0,out_dim=1,
                conditioning='cross_attention',norm_eps=1e-5,key=jax.random.key(1))
            image=jnp.linspace(-1,1,64).reshape(1,4,4,4)
            x=jnp.array([[.2,.3,.1],[.6,.8,.5],[.4,.5,.7]])
            times=jnp.array([0.,.4,.8]);local=jnp.zeros((3,0))
            raw=net(image,x,times,local)
            paired=net.paired(image,jnp.tile(x,(3,1)),jnp.repeat(times,3),jnp.zeros((9,0))).reshape(raw.shape)
            np.testing.assert_allclose(raw,paired,rtol=1e-12,atol=1e-12)
            initial=jnp.array([-.7,.3,.8]);encoded=net.encode(image)
            u,ut=latent_fields(net,encoded,x,initial,times)
            np.testing.assert_array_equal(u[0],initial)
            h=1e-5
            f=lambda ts:initial[None,:]+ts[:,None]*net(image,x,ts,local)[...,0]
            finite=(f(times+h)-f(times-h))/(2*h)
            np.testing.assert_allclose(ut,finite,rtol=2e-7,atol=1e-9)
            jac=jax.jacfwd(f)(times)
            np.testing.assert_allclose(jac[0,:,1:],0.,atol=1e-14)
            row=jnp.array([0,1,1,2]);col=jnp.array([1,0,2,1]);w=jnp.full(4,2.)
            sub=(row,col,-w,jnp.array([2.,4.,2.]));mask=jnp.array([True,True,False])
            loss=make_loss_fn(c)
            objective=eqx.filter_jit(lambda n:loss(n,times,image[None],initial[None],x,sub,mask))
            phi=np.tanh(np.asarray(u));A=np.array([[2.,-2.,0.],[-2.,4.,-2.],[0.,-2.,2.]])
            n1,n2,n3,_=physics(c)
            physical=(1-phi**2)*np.asarray(ut)+n1*(A@phi.T).T+n2*(phi**3-phi)+n3*(1-phi**2)
            expected=np.mean(abs((physical/(1-phi**2))[:,:2]))
            np.testing.assert_allclose(objective(net),expected,rtol=2e-13)
            tangent=eqx.filter_grad(objective)(net)
            self.assertGreater(np.linalg.norm(np.asarray(tangent.patch_projection.weight)),1e-8)
            direction=jnp.ones_like(net.patch_projection.weight)/net.patch_projection.weight.size**.5
            plus=eqx.tree_at(lambda n:n.patch_projection.weight,net,net.patch_projection.weight+h*direction)
            minus=eqx.tree_at(lambda n:n.patch_projection.weight,net,net.patch_projection.weight-h*direction)
            fd=(objective(plus)-objective(minus))/(2*h)
            np.testing.assert_allclose(jnp.sum(tangent.patch_projection.weight*direction),fd,rtol=2e-5,atol=2e-8)
            np.testing.assert_array_equal(tangent.spatial_frequencies,0.)

    def test_independent_reference_stencil_matches_actual_graph(self):
        with jax.enable_x64(True):
            c=get_config();c.sample.Nx=c.sample.Ny=c.sample.Nz=8;c.nl.delta=1.27/7
            axis=np.linspace(0,1,8);points=np.stack(np.meshgrid(axis,axis,axis,indexing='ij'),-1).reshape(-1,3)
            graph=build_D(jnp.asarray(points),c.nl.delta,method='sparse',kernel='cubic')
            field=np.random.default_rng(3).uniform(-.9,.9,(8,8,8))
            n1,n2,n3,weight=coefficients(c)
            expected=-n1*np.asarray((graph @ jnp.asarray(field.ravel()))).reshape(field.shape)-n2*(field**3-field)-n3*(1-field**2)
            np.testing.assert_allclose(rhs(jnp.asarray(field),c),expected,rtol=2e-12,atol=2e-12)
            np.testing.assert_allclose(np.asarray(graph._vals),-weight,rtol=1e-12)


if __name__=='__main__':unittest.main()
