"""Integral moment accuracy, truncated supports and strict payload contracts."""
import unittest

import numpy as np
from scipy.spatial import cKDTree

from examples.channel_nsch2d.configs.train_operator_width128_integral_time40_jvp_stronger_online import get_config
from examples.channel_nsch2d.integral_operators import integral_stencils
from examples.channel_nsch2d.problem import boundary,quadrature,validate_config


class IntegralOperatorsTest(unittest.TestCase):
    def test_boundary_polynomials_and_independent_constrained_solve(self):
        c=get_config();c.sample.Nx=48;c.sample.Ny=20;c.nl.delta=.3
        x,v=quadrature((.18,.12),c,48,20)
        b,_=boundary((.18,.12),c)
        target=np.concatenate((x[[0,20,-1]],b[::17,:2]))
        ids,gx,gy,lap,report=integral_stencils(x,v,target,c)
        self.assertLess(report['polynomial_error'],5e-9)
        for a in range(4):
            for d in range(4-a):
                f=x[:,0]**a*x[:,1]**d;center=target[:,0]**a*target[:,1]**d
                adjacent=f[ids]-center[:,None]
                expected_x=(a*target[:,0]**(a-1)*target[:,1]**d if a else np.zeros(len(target)))
                expected_y=(d*target[:,0]**a*target[:,1]**(d-1) if d else np.zeros(len(target)))
                expected_lap=(a*(a-1)*target[:,0]**(a-2)*target[:,1]**d if a>1 else np.zeros(len(target)))
                expected_lap+=(d*(d-1)*target[:,0]**a*target[:,1]**(d-2) if d>1 else np.zeros(len(target)))
                for weights,truth in ((gx,expected_x),(gy,expected_y),(lap,expected_lap)):
                    np.testing.assert_allclose(np.sum(weights*adjacent,axis=1),truth,atol=2e-9,rtol=0)
        # Independent normal-equation evaluation at a truncated corner; it
        # verifies the correction, not just polynomial exactness of any stencil.
        xi=x[ids[0]]-target[0];r=np.linalg.norm(xi,axis=1)
        active=np.arange(c.nl.neighbors)<len(cKDTree(x).query_ball_point(target[0],c.nl.delta))
        xi=xi[active];rows=ids[0,active];z=xi/c.nl.delta
        B=np.column_stack((z[:,0],z[:,1],z[:,0]**2,z[:,0]*z[:,1],z[:,1]**2,
                           z[:,0]**3,z[:,0]**2*z[:,1],z[:,0]*z[:,1]**2,z[:,1]**3))
        w=120/(np.pi*c.nl.delta**4)*(1-r[active]/c.nl.delta)**2*v[rows]
        rhs=np.array([0,0,2/c.nl.delta**2,0,2/c.nl.delta**2,0,0,0,0])
        independent=w+w*(B@np.linalg.solve(B.T@(w[:,None]*B),rhs-B.T@w))
        np.testing.assert_allclose(lap[0,active],independent,atol=2e-8,rtol=2e-10)

    def test_smooth_refinement_and_no_slip_wall_layer(self):
        errors=[]
        for nx,ny,delta in ((48,20,.3),(96,40,.15),(192,80,.075)):
            c=get_config();c.sample.Nx=nx;c.sample.Ny=ny;c.nl.delta=delta
            x,v=quadrature((.18,.12),c,nx,ny)
            ids,gx,gy,lap,_=integral_stencils(x,v,x,c)
            f=np.exp(.2*x[:,0]+.3*x[:,1]);df=f[ids]-f[:,None]
            actual=np.column_stack((np.sum(gx*df,axis=1),np.sum(gy*df,axis=1),np.sum(lap*df,axis=1)))
            truth=f[:,None]*np.array([.2,.3,.13])
            errors.append(np.linalg.norm(actual-truth,axis=0)/np.linalg.norm(truth,axis=0))
            u=6*c.physical.mean_velocity*x[:,1]*(1-x[:,1])
            np.testing.assert_allclose(np.sum(lap*(u[ids]-u[:,None]),axis=1),
                -12*c.physical.mean_velocity,atol=2e-9,rtol=0)
        errors=np.asarray(errors)
        self.assertTrue(np.all(errors[1:]<errors[:-1]/2),errors)

    def test_invalid_support_capacity_and_configuration_fail(self):
        c=get_config();validate_config(c)
        x,v=quadrature((.18,.12),c,48,20)
        c.nl.delta=.001
        with self.assertRaisesRegex(ValueError,'unresolved'):
            integral_stencils(x,v,x,c)
        c.nl.delta=.6;c.nl.neighbors=10
        with self.assertRaisesRegex(ValueError,'capacity'):
            integral_stencils(x,v,x,c)
        c=get_config();c.derivatives.boundary='gmls'
        with self.assertRaises(ValueError):validate_config(c)

    def test_full_time_jvp_loss_gradient_against_finite_difference(self):
        import equinox as eqx
        import jax
        import jax.numpy as jnp
        from examples.channel_nsch2d.data import geometry_data
        from examples.channel_nsch2d.losses import make_loss_fn
        from examples.channel_nsch2d.network import build_network
        c=get_config();c.train.compute_dtype='float64';c.train.seed=42
        c.sample.Nx=24;c.sample.Ny=12;c.sample.Nt=2;c.sample.boundary_points=6
        c.nl.delta=.6;c.nl.neighbors=192;c.train.microbatch.size=1
        c.arch.width=4;c.arch.heads=1;c.arch.condition_depth=1;c.arch.decoder_depth=1
        c.arch.spatial_features=2;c.arch.time_features=4
        with jax.enable_x64(True):
            raw,_=geometry_data((.18,.12),c)
            data={k:jnp.asarray(v)[None]for k,v in raw.items()}
            net=build_network(c,jax.random.PRNGKey(8));times=jnp.array([.25,1.])
            loss=make_loss_fn(c)
            (value,parts),gradient=eqx.filter_jit(eqx.filter_value_and_grad(loss,has_aux=True))(net,data,times)
            self.assertEqual(parts.shape,(9,));self.assertTrue(np.isfinite(value))
            epsilon=2e-5;index=(0,0)
            evaluate=eqx.filter_jit(lambda n:loss(n,data,times)[0])
            plus=eqx.tree_at(lambda n:n.output.weight,net,net.output.weight.at[index].add(epsilon))
            minus=eqx.tree_at(lambda n:n.output.weight,net,net.output.weight.at[index].add(-epsilon))
            numeric=float((evaluate(plus)-evaluate(minus))/(2*epsilon))
            self.assertGreater(abs(numeric),1e-8)
            np.testing.assert_allclose(gradient.output.weight[index],numeric,rtol=3e-5,atol=3e-7)


if __name__=='__main__':unittest.main()
