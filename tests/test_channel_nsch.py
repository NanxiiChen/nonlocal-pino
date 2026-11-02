"""Physical geometry, derivative consistency, exact constraints and NS--CH signs."""
import unittest
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from examples.channel_nsch2d.configs.train_base import get_config
from examples.channel_nsch2d.problem import walls,flow_rate,initial_field,validate_config
from examples.channel_nsch2d.data import geometry_data
from examples.channel_nsch2d.operators import laplacian_stencils
from examples.channel_nsch2d.network import build_network,encode,fields,field_jets
from examples.channel_nsch2d.losses import residual_components,make_loss_fn,part_names


def small():
    c=get_config();c.sample.Nx=8;c.sample.Ny=4;c.sample.boundary_points=6
    c.nl.neighbors=20;c.nl.degree=2
    c.arch.width=8;c.arch.heads=2;c.arch.condition_depth=1;c.arch.decoder_depth=1
    c.arch.spatial_features=2;c.arch.time_features=2;c.train.compute_dtype='float64'
    data,_=geometry_data((.12,.08),c)
    data={k:jnp.asarray(v) for k,v in data.items()}
    return c,data,build_network(c,jax.random.PRNGKey(42))


class ChannelTests(unittest.TestCase):
    def test_invalid_configuration_fails_early(self):
        for field,value in [('phase_scale',0.),('chemical_scale',float('nan'))]:
            c=get_config();setattr(c.loss,field,value)
            with self.assertRaisesRegex(ValueError,'scales'):validate_config(c)
        c=get_config();c.lifting.time_scale=float('nan')
        with self.assertRaisesRegex(ValueError,'lifting'):validate_config(c)
        c=get_config();c.operator.condition_batch=0
        with self.assertRaisesRegex(ValueError,'counts'):validate_config(c)

    def test_geometry_and_compact_initial(self):
        c=get_config();validate_config(c)
        x=np.linspace(0,4,101);b,t=walls(x,(.14,.04),c,xp=np)
        self.assertTrue(np.all(t>b));self.assertAlmostEqual(t[50]-b[50],.82)
        self.assertAlmostEqual((t[50]+b[50])/2,.45)
        data,_=geometry_data((.14,.04),c)
        np.testing.assert_allclose(initial_field(data['boundary'][:96,:2],c,xp=np),1.,atol=0,rtol=0)
        self.assertEqual(flow_rate(0.,c,xp=np),0.)
        self.assertAlmostEqual(flow_rate(2.,c,xp=np),.2)

    def test_polynomial_reproduction_and_smooth_refinement(self):
        c=get_config();errors=[]
        for nx,ny in ((16,8),(32,16)):
            c.sample.Nx=nx;c.sample.Ny=ny
            data,r=geometry_data((.12,.08),c);x=data['points'];idx=data['neighbors'];w=data['laplacian']
            for f,target in ((np.ones(len(x)),0.),(x[:,0],0.),(x[:,1],0.),
                             (x[:,0]**2+x[:,1]**2,4.),(x[:,0]*x[:,1],0.)):
                np.testing.assert_allclose(np.sum(w*(f[idx]-f[:,None]),axis=1),target,atol=2e-10)
            f=np.sin(x[:,0])*np.cos(x[:,1]);actual=np.sum(w*(f[idx]-f[:,None]),axis=1)
            mask=data['interior']>0
            errors.append(np.linalg.norm((actual+2*f)[mask])/np.linalg.norm(2*f[mask]))
        self.assertLess(errors[1],errors[0]*.6)
        self.assertLess(errors[1],.08)

    def test_momentum_and_coupling_signs(self):
        with jax.enable_x64(True):
            c=get_config();x=jnp.array([[.4,.2],[1.,.6],[3.,.8]]);y=x[:,1];Q=.2
            value=jnp.stack((jnp.ones(3),jnp.zeros(3),6*Q*y*(1-y),jnp.zeros(3),
                             12*c.physical.viscosity*Q*(4-x[:,0])),axis=-1)
            zero=jnp.zeros_like(value);dx=zero.at[:,4].set(-12*c.physical.viscosity*Q)
            dy=zero.at[:,2].set(6*Q*(1-2*y));lap=zero.at[:,2].set(-12*Q)
            np.testing.assert_allclose(residual_components(value,zero,dx,dy,lap,c),0.,atol=1e-14)
            # Independent capillary-force perturbation must enter momentum +phi*grad(mu).
            dx=dx.at[:,1].set(.7)
            np.testing.assert_allclose(residual_components(value,zero,dx,dy,lap,c)[:,2],.7,atol=1e-14)

    def test_exact_initial_inlet_wall_and_query_jvp(self):
        with jax.enable_x64(True):
            c,d,net=small();e=encode(net,d);times=jnp.array([0.,.6,2.])
            b=d['boundary'];v=fields(net,e,b[:,:2],times,d['amplitudes'],c)
            np.testing.assert_allclose(np.asarray(v)[:,np.asarray(b[:,4])==0,2:4],0.,atol=2e-15)
            np.testing.assert_allclose(np.asarray(v)[0,:,0],initial_field(np.asarray(b[:,:2]),c,xp=np),atol=0,rtol=0)
            np.testing.assert_allclose(np.asarray(v)[0,:,2:4],0.,atol=0,rtol=0)
            mask=np.asarray(b[:,4])==1;y=np.asarray(b[mask,1])
            expected=6*np.asarray(flow_rate(times,c,xp=jnp))[:,None]*y*(1-y)
            np.testing.assert_allclose(np.asarray(v)[:,mask,2],expected,atol=2e-15)
            np.testing.assert_allclose(np.asarray(v)[:,mask,1],0.,atol=0,rtol=0)
            x=jnp.array([[.8,.4],[2.,.6]])
            _,dt,dx,dy=field_jets(net,e,x,times,d['amplitudes'],c)
            for i in range(len(x)):
                derivative=jax.jacfwd(lambda point:fields(net,e,point[None],times,d['amplitudes'],c)[:,0,:])(x[i])
                np.testing.assert_allclose(dx[:,i,:],derivative[...,0],atol=2e-12)
                np.testing.assert_allclose(dy[:,i,:],derivative[...,1],atol=2e-12)
            np.testing.assert_allclose(dx[...,2]+dy[...,3],0.,atol=2e-12)
            jac=jax.jacfwd(lambda ts:fields(net,e,x,ts,d['amplitudes'],c))(times)
            np.testing.assert_allclose(dt,jnp.sum(jac,axis=-1),atol=2e-12)

    def test_streamfunction_flux_is_exact_for_every_section(self):
        with jax.enable_x64(True):
            c,d,net=small();e=encode(net,d)
            roots,weights=np.polynomial.legendre.leggauss(40)
            eta=(roots+1)/2;weights=weights/2
            section=np.array([0.,.03,.8,2.,3.9,4.])
            xx,ee=np.meshgrid(section,eta,indexing='ij')
            b,t=walls(xx,(.12,.08),c,xp=np)
            points=jnp.asarray(np.stack((xx,b+ee*(t-b)),axis=-1).reshape(-1,2))
            times=jnp.array([0.,.4,2.])
            value=fields(net,e,points,times,d['amplitudes'],c)
            flux=np.sum(np.asarray(value[...,2]).reshape(3,len(section),40)*(t-b)[None]*weights,axis=-1)
            np.testing.assert_allclose(flux,np.broadcast_to(np.asarray(flow_rate(times,c,xp=jnp))[:,None],flux.shape),atol=2e-11)

    def test_complete_differentiated_residual_is_finite(self):
        with jax.enable_x64(True):
            c,d,net=small();batch=jax.tree.map(lambda a:a[None],d)
            fn=make_loss_fn(c)
            (total,parts),g=eqx.filter_jit(eqx.filter_value_and_grad(fn,has_aux=True))(
                net,batch,jnp.array([.3,1.]))
            self.assertTrue(np.isfinite(float(total)));self.assertEqual(parts.shape,(len(part_names(c)),))
            leaves=jax.tree.leaves(eqx.filter(g,eqx.is_array))
            self.assertTrue(all(np.isfinite(np.asarray(a)).all() for a in leaves))
            self.assertGreater(sum(float(jnp.sum(a*a)) for a in leaves),0.)


if __name__=='__main__':unittest.main()
