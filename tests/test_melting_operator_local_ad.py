"""Independent known-geometry, local jets and transformed-PDE checks."""
import unittest

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from examples.icemelting3d.ellipsoids import signed_distance_numpy
from examples.icemelting3d.ellipsoid_jets import signed_distance_jets
from examples.icemelting3d.operator_data import build_network, physics
from examples.icemelting3d.operator_local_ad import (
    correction, local_fields, make_loss_fn, sample_boundary, stable_tanh_jacobian)
from examples.icemelting3d.configs.train_operator_compare_local_ad import get_config


class Polynomial(eqx.Module):
    coefficient: jax.Array

    def encode(self, image):
        return image

    def prepare_encoded(self, encoded, points, local):
        return points

    def decode(self, points, times):
        x, y, z = points.T
        return (self.coefficient*(x*x+2*y*y+3*z*z)[None, :]
                +times[:, None]*(x*y*z)[None, :])[..., None]


class MeltingLocalAD(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_x64 = jax.config.jax_enable_x64
        jax.config.update('jax_enable_x64', True)

    @classmethod
    def tearDownClass(cls):
        jax.config.update('jax_enable_x64', cls.previous_x64)

    def test_sphere_geometry_and_medial_axis(self):
        p = np.array([.5, .5, .5, .3, .3, .3, 1., 0., 0., 0.])
        x = np.array([[.65, .55, .52], [.2, .3, .4], [.8, .5, .5]])
        gradient, lap, valid = signed_distance_jets(jnp.asarray(x), jnp.asarray(p), 56)
        d = x-.5; r = np.linalg.norm(d, axis=1)
        np.testing.assert_allclose(gradient, -d/r[:, None], rtol=2e-12, atol=2e-12)
        np.testing.assert_allclose(lap, -2/r, rtol=2e-12, atol=2e-12)
        self.assertTrue(np.asarray(valid).all())
        self.assertFalse(bool(signed_distance_jets(jnp.array([[.5,.5,.5]]), jnp.asarray(p), 56)[2][0]))

    def test_rotated_triaxial_geometry_against_independent_fd(self):
        p = np.array([.48,.52,.5,.27,.31,.35,.8,.2,-.3,.1]); p[6:] /= np.linalg.norm(p[6:])
        x = np.array([[.62,.55,.57], [.7,.6,.66], [.15,.2,.3], [.5,.54,.51], [.46,.8,.5]])
        gradient, lap, valid = signed_distance_jets(jnp.asarray(x), jnp.asarray(p), 56)
        self.assertTrue(np.asarray(valid).all())
        h = 2e-5; value = signed_distance_numpy(x, p, 64)
        fd_gradient, fd_lap = [], np.zeros(len(x))
        for axis in np.eye(3):
            plus, minus = signed_distance_numpy(x+h*axis, p, 64), signed_distance_numpy(x-h*axis, p, 64)
            fd_gradient.append((plus-minus)/(2*h)); fd_lap += (plus-2*value+minus)/h**2
        np.testing.assert_allclose(gradient, np.array(fd_gradient).T, rtol=2e-6, atol=2e-6)
        np.testing.assert_allclose(lap, fd_lap, rtol=8e-6, atol=2e-5)
        g32, l32, v32 = signed_distance_jets(jnp.asarray(x,dtype=jnp.float32), jnp.asarray(p,dtype=jnp.float32), 40)
        self.assertTrue(np.asarray(v32).all())
        np.testing.assert_allclose(g32, gradient, rtol=2e-5, atol=2e-5)
        np.testing.assert_allclose(l32, lap, rtol=4e-5, atol=4e-5)

    def test_polynomial_pde_chain_rule_and_parameter_gradient(self):
        c = get_config(); c.train.seed = 42
        c.operator.local_ad.boundary_points = 6
        net = Polynomial(jnp.asarray(.7))
        x = jnp.array([[.2,.3,.4],[.6,.5,.2]])
        t = jnp.array([.15,.8]); u0 = jnp.sum(x*x, axis=1)
        g0, l0 = 2*x, jnp.full((len(x),), 6.)
        boundary, normals = (jnp.asarray(a, dtype=jnp.float64) for a in sample_boundary(c, 0))
        b0, bg0 = jnp.sum(boundary**2,axis=1), 2*boundary
        jets = (g0[None], l0[None], b0[None], bg0[None], jnp.array([True]))
        loss = make_loss_fn(c)
        value, parts = loss(net,t,jnp.zeros((1,1)),u0[None],x,jets,boundary,normals)
        n1,n2,n3,_ = physics(c)

        def physical(point, time, a):
            xx,yy,zz = point
            return jnp.tanh(jnp.sum(point**2)+time*(a*(xx*xx+2*yy*yy+3*zz*zz)+time*xx*yy*zz))

        def independent(a):
            residuals=[]; fluxes=[]
            for time in t:
                for point in x:
                    phi=physical(point,time,a)
                    dt=jax.grad(lambda tt:physical(point,tt,a))(time)
                    lap=jnp.trace(jax.hessian(lambda pp:physical(pp,time,a))(point))
                    residuals.append((dt-n1*lap+n2*(phi**3-phi)+n3*(1-phi**2))/(1-phi**2))
                for point,normal in zip(boundary,normals,strict=True):
                    fluxes.append(jnp.dot(jax.grad(lambda pp:physical(pp,time,a))(point),normal))
            return jnp.mean(jnp.abs(jnp.array(residuals)))+jnp.mean(jnp.array(fluxes)**2)

        expected=independent(net.coefficient)
        np.testing.assert_allclose(value,expected,rtol=2e-12,atol=2e-12)
        grad=eqx.filter_grad(lambda n:loss(n,t,jnp.zeros((1,1)),u0[None],x,jets,boundary,normals)[0])(net)
        np.testing.assert_allclose(grad.coefficient,jax.grad(independent)(net.coefficient),rtol=2e-10,atol=2e-11)
        self.assertTrue(np.all(np.asarray(parts)>0))

    def test_actual_attention_query_jets_are_pointwise(self):
        c=get_config(); c.arch.image_shape=(1,4,4,4); c.arch.patch_shape=(2,2,2)
        c.arch.width=12; c.arch.heads=2; c.arch.encoder_depth=1; c.arch.decoder_depth=1
        c.arch.spatial_features=4; c.arch.time_features=4
        net=build_network(c,jax.random.key(7))
        net=eqx.tree_at(lambda n:n.output.weight,net,jnp.ones_like(net.output.weight)*.2)
        encoded=net.encode(jnp.arange(64,dtype=jnp.float64).reshape(1,4,4,4)/64)
        x=jnp.array([[.23,.47,.67],[.36,.55,.42]]); t=jnp.array([.1,.6])
        u,ut,g,lap=local_fields(net,encoded,x,t,jnp.zeros(2),jnp.zeros((2,3)),jnp.zeros(2))
        for i,tt in enumerate(t):
            for j,xx in enumerate(x):
                scalar=lambda point,time:time*correction(net,encoded,point[None],time[None])[0,0]
                np.testing.assert_allclose(u[i,j],scalar(xx,tt),rtol=2e-11,atol=2e-11)
                np.testing.assert_allclose(ut[i,j],jax.grad(lambda time:scalar(xx,time))(tt),rtol=2e-10,atol=2e-10)
                np.testing.assert_allclose(g[i,j],jax.grad(lambda point:scalar(point,tt))(xx),rtol=2e-9,atol=2e-9)
                np.testing.assert_allclose(lap[i,j],jnp.trace(jax.hessian(lambda point:scalar(point,tt))(xx)),rtol=2e-8,atol=2e-8)

    def test_boundary_stream_and_saturated_jacobian(self):
        c=get_config(); c.train.seed=42
        a,n=sample_boundary(c,0); b,m=sample_boundary(c,1)
        self.assertFalse(np.array_equal(a,b)); np.testing.assert_array_equal(n,m)
        self.assertEqual(len(a),96)
        np.testing.assert_array_equal(np.sum(n*n,axis=1),1)
        np.testing.assert_array_equal(np.sum(a*n,axis=1),np.where(n.sum(axis=1)>0,1,0))
        u=jnp.array([-20.,-2.,0.,2.,20.],dtype=jnp.float32)
        expected=1/np.cosh(np.asarray(u,dtype=np.float64))**2
        np.testing.assert_allclose(stable_tanh_jacobian(u),expected,rtol=3e-7,atol=0)

    def test_cold_step_and_native_sampling_identity(self):
        from examples.icemelting3d.operator_data import make_sampler, identity, validate_config
        from examples.icemelting3d.operator_local_ad import make_initial_jets, make_step
        from examples.icemelting3d.configs.train_operator_compare_nonlocal import get_config as nonlocal_config
        from nonlocal_pinn.training.optimizers import make_optimizer
        c=get_config(); c.train.seed=42; validate_config(c)
        self.assertEqual(identity(c),identity(nonlocal_config()))
        c.arch.image_shape=(1,4,4,4); c.arch.patch_shape=(2,2,2)
        c.arch.width=12; c.arch.heads=2; c.arch.encoder_depth=1; c.arch.decoder_depth=1
        c.arch.spatial_features=4; c.arch.time_features=4; c.operator.condition_batch=2
        c.operator.local_ad.boundary_points=6
        x=jnp.array([[.2,.3,.4],[.6,.5,.2]],dtype=jnp.float32)
        t=jnp.array([.2,.7],dtype=jnp.float32)
        sampler=make_sampler(c); key=jax.random.key(12)
        images,u0,parameters=sampler(key,x)
        expanded=jnp.concatenate((x,jnp.array([[.9,.8,.1]],dtype=jnp.float32)))
        images2,u02,parameters2=sampler(key,expanded)
        np.testing.assert_array_equal(images,images2); np.testing.assert_array_equal(parameters,parameters2)
        np.testing.assert_allclose(u0,u02[:,:2],rtol=0,atol=2e-6)
        boundary,normals=map(jnp.asarray,sample_boundary(c,0))
        jets=make_initial_jets(c)(x,boundary,parameters)
        self.assertTrue(np.asarray(jets[-1]).all())
        net=build_network(c,jax.random.key(8)); optimizer=make_optimizer(c)
        state=optimizer.init(eqx.filter(net,eqx.is_array))
        updated,_,loss,finite,parts=make_step(c,optimizer)(net,state,t,images,u0,x,jets,boundary,normals)
        self.assertTrue(bool(finite)); self.assertTrue(np.isfinite(float(loss)))
        self.assertEqual(parts.shape,(2,))
        self.assertFalse(np.array_equal(net.output.bias,updated.output.bias))


if __name__ == '__main__':
    unittest.main()
