"""Published SBP identity, mapped refinement and open-boundary conservation."""
import unittest
import numpy as np
from examples.channel_nsch2d.configs.train_operator import get_config
from examples.channel_nsch2d.problem import quadrature,walls,wall_slopes
from examples.channel_nsch2d.transport import sbp_first,build_transport,conservative_advection


class ChannelTransport(unittest.TestCase):
    def test_float32_training_conservation_and_full_mixed_gradient(self):
        import equinox as eqx
        import jax
        import jax.numpy as jnp
        from examples.channel_nsch2d.data import geometry_data
        from examples.channel_nsch2d.configs.train_single_conservative import get_config as configured
        from examples.channel_nsch2d.losses import make_loss_fn,residual_components
        from examples.channel_nsch2d.problem import validate_config
        from tests.test_channel_nsch import small
        c=configured();validate_config(c);a=np.array([.12,.08]);points,_=quadrature(a,c,96,40)
        data=build_transport(points,a,c);data={k:jnp.asarray(v,dtype=jnp.float32) for k,v in data.items()}
        eta=np.linspace(0,1,41)[None,:];xx=points.reshape(97,41,2)[...,0]
        psi=.2*(3*eta**2-2*eta**3)+.01*np.sin(xx)*16*eta**2*(1-eta)**2
        phi=jnp.asarray(np.random.default_rng(91).normal(size=(1,len(points))),dtype=jnp.float32)
        f=jax.jit(lambda p,s:conservative_advection(p,s,data,xp=jnp))
        rate,qx,qe=[np.asarray(v,dtype=np.float64) for v in f(phi,jnp.asarray(psi.reshape(1,-1),dtype=jnp.float32))]
        p=np.asarray(phi).reshape(97,41);Hx=np.asarray(data['transport_weight_x']);He=np.asarray(data['transport_weight_eta'])
        boundary=np.sum(He*(p[-1]*qx[0,-1]-p[0]*qx[0,0]))+np.sum(Hx*(p[:,-1]*qe[0,:,-1]-p[:,0]*qe[0,:,0]))
        self.assertLess(abs(float(rate[0]@np.asarray(data['transport_measure'])-boundary)),3e-6)
        with jax.enable_x64(True):
            c,_,net=small();c.sample.Ny=8;c.sample.Nt=2
            c.phase_transport=configured().phase_transport;c.time_discretization=configured().time_discretization
            raw,_=geometry_data(a,c);data={k:jnp.asarray(v) for k,v in raw.items()}
            batch=jax.tree.map(lambda v:v[None],data);t=jnp.array([.05,2.])
            (loss,parts),gradient=eqx.filter_jit(eqx.filter_value_and_grad(make_loss_fn(c),has_aux=True))(net,batch,t)
            self.assertEqual(parts.shape,(9,));self.assertTrue(np.isfinite(float(loss)))
            self.assertTrue(all(np.isfinite(np.asarray(v)).all() for v in jax.tree.leaves(gradient) if eqx.is_array(v)))
            v=jnp.ones((2,3,5));zero=jnp.zeros_like(v);replacement=jnp.full((2,3),.7)
            before=residual_components(v,zero,zero,zero,zero,c)
            after=residual_components(v,zero,zero,zero,zero,c,phase_advection=replacement)
            np.testing.assert_allclose(after[...,0]-before[...,0],.7,atol=1e-14)
            np.testing.assert_array_equal(after[...,1:],before[...,1:])

    def test_sbp_polynomials_quadrature_and_boundary_identity(self):
        for n in (9,17,33):
            x=np.linspace(0,4,n);D,H=sbp_first(n,4.)
            for p in (0,1,2):
                expected=np.zeros(n) if p==0 else p*x**(p-1)
                np.testing.assert_allclose(D@x**p,expected,atol=2e-12,rtol=0)
            for p in (3,4):np.testing.assert_allclose((D@x**p)[4:-4],(p*x**(p-1))[4:-4],atol=2e-11,rtol=0)
            self.assertAlmostEqual(float(H.sum()),4.,places=14);self.assertTrue(np.all(H>0))
            u=np.sin(x);v=np.exp(.2*x)
            self.assertAlmostEqual(float(u@(H*(D@v))+(D@u)@(H*v)),float(u[-1]*v[-1]-u[0]*v[0]),places=13)

    def test_mapped_wall_divergence_and_arbitrary_phase_boundary_flux(self):
        c=get_config();c.sample.Nx=24;c.sample.Ny=12;a=np.array([.12,.08])
        points,_=quadrature(a,c,24,12);d=build_transport(points,a,c)
        xx=points.reshape(25,13,2)[...,0];eta=np.linspace(0,1,13)[None,:]
        stream=.2*(3*eta**2-2*eta**3)+.01*np.sin(xx)*16*eta**2*(1-eta)**2
        phi=np.random.default_rng(32).normal(size=(1,len(points)))
        rate,qx,qe=conservative_advection(phi,stream.reshape(1,-1),d,xp=np)
        constant,_,_=conservative_advection(np.ones_like(phi),stream.reshape(1,-1),d,xp=np)
        np.testing.assert_allclose(constant,0.,atol=1e-12)
        np.testing.assert_allclose(qe[0,:,[0,-1]],0.,atol=2e-14)
        p=phi.reshape(1,25,13)
        boundary=np.sum(d['transport_weight_eta']*(p[0,-1]*qx[0,-1]-p[0,0]*qx[0,0]))
        boundary+=np.sum(d['transport_weight_x']*(p[0,:,-1]*qe[0,:,-1]-p[0,:,0]*qe[0,:,0]))
        self.assertAlmostEqual(float(rate[0]@d['transport_measure']),float(boundary),places=13)
        with self.assertRaises(AssertionError):build_transport(points[::-1],a,c)

    def test_manufactured_physical_advection_refines_on_curved_channel(self):
        c=get_config();a=np.array([.12,.08]);errors=[]
        for nx,ny in ((24,12),(48,24),(96,48)):
            c.sample.Nx=nx;c.sample.Ny=ny;x,_=quadrature(a,c,nx,ny);d=build_transport(x,a,c)
            xx,yy=x[:,0],x[:,1];bottom,top=walls(xx,a,c,xp=np)
            eta=(yy-bottom)/(top-bottom);db,dt=wall_slopes(xx,a,c,xp=np)
            slope=db+eta*(dt-db);ux=.2*6*eta*(1-eta)/(top-bottom);uy=slope*ux
            phase=np.sin(xx)*np.cos(yy);stream=.2*(3*eta**2-2*eta**3)
            exact=ux*np.cos(xx)*np.cos(yy)-uy*np.sin(xx)*np.sin(yy)
            actual,_,_=conservative_advection(phase[None],stream[None],d,xp=np)
            w=d['transport_measure'];errors.append(np.sqrt(np.sum((actual[0]-exact)**2*w)/np.sum(exact**2*w)))
        self.assertLess(errors[1],errors[0]*.3);self.assertLess(errors[2],errors[1]*.3)
        self.assertLess(errors[-1],.002)


if __name__=='__main__':unittest.main()
