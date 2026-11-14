"""Independent mixed FE quadrature, boundary signs and differentiated loss checks."""
import unittest
import numpy as np

from examples.channel_nsch2d.configs.train_operator_fem_weak import get_config
from examples.channel_nsch2d.fem_weak import basis, build, make_weak_rows, triangle_rule
from examples.channel_nsch2d.problem import material, validate_config


def independent_basis(q):
    nodes = np.array([[0,0],[1,0],[0,1],[.5,0],[.5,.5],[0,.5]])
    polynomial = lambda a: np.array([np.ones(len(a)),a[:,0],a[:,1],a[:,0]**2,a[:,0]*a[:,1],a[:,1]**2]).T
    coefficient = np.linalg.inv(polynomial(nodes))
    x,y = np.asarray(q).T
    dx = np.array([np.zeros(len(q)),np.ones(len(q)),np.zeros(len(q)),2*x,y,np.zeros(len(q))]).T
    dy = np.array([np.zeros(len(q)),np.zeros(len(q)),np.ones(len(q)),np.zeros(len(q)),x,2*y]).T
    return polynomial(q)@coefficient, np.stack((dx@coefficient,dy@coefficient),axis=-1)


def fixture():
    c=get_config();c.sample.Nx=8;c.sample.Ny=4;c.train.compute_dtype='float64'
    x,y=np.meshgrid(np.linspace(0,4,9),np.linspace(0,1,5),indexing='ij')
    points=np.stack((x,y),axis=-1).reshape(-1,2)
    data,report=build(points,c)
    return c,points,data,report


def oracle(value,rate,points,data,c):
    """Scalar element loop, independent Vandermonde basis and higher quadrature."""
    q,w=triangle_rule(7);N,dN=independent_basis(q)
    A,kappa=material(c);p=c.physical
    out=np.zeros((len(points),4))
    for cell in data['fem_cells']:
        for z,wt,n,dn in zip(q,w,N,dN,strict=True):
            jac=points[cell].T@dn;measure=np.linalg.det(jac)*wt
            g=dn@np.linalg.inv(jac)
            v=n@value[cell];t=n@rate[cell];grad=value[cell].T@g
            phi,mu,ux,uy=v[:4];pr=np.array([1-z.sum(),*z])@value[cell[:3],4]
            load=np.array([t[0],mu-A*(phi**3-phi),
                p.density*(t[2]+ux*grad[2,0]+uy*grad[2,1])+phi*grad[1,0],
                p.density*(t[3]+ux*grad[3,0]+uy*grad[3,1])+phi*grad[1,1]])
            shear=p.viscosity*(grad[2,1]+grad[3,0])
            flux=np.array([[-phi*ux+p.mobility*grad[1,0],-phi*uy+p.mobility*grad[1,1]],
                -kappa*grad[0], [2*p.viscosity*grad[2,0]-pr,shear],
                [shear,2*p.viscosity*grad[3,1]-pr]])
            np.add.at(out,cell,measure*(n[:,None]*load+g@flux.T))
    z,w=np.polynomial.legendre.leggauss(7)
    for cell in data['fem_outlet']:
        for s,wt in zip((z+1)/2,w/2,strict=True):
            n=np.array([(1-s)*(1-2*s),s*(2*s-1),4*s*(1-s)])
            dn=np.array([4*s-3,4*s-1,4-8*s]);tangent=dn@points[cell]
            v=n@value[cell];flux=v[0]*np.dot(v[2:4],[tangent[1],-tangent[0]])
            np.add.at(out[:,0],cell,wt*flux*n)
    return out


class ChannelFemTest(unittest.TestCase):
    def test_basis_geometry_and_payload(self):
        c,x,d,r=fixture();validate_config(c)
        q,_=triangle_rule(6);n,g,_=basis(q);on,og=independent_basis(q)
        np.testing.assert_allclose(n,on,atol=1e-15);np.testing.assert_allclose(g,og,atol=2e-15)
        self.assertAlmostEqual(r['area'],4.,places=13)
        self.assertGreater(d['fem_mass_diagonal'].min(),0)
        np.testing.assert_array_equal(np.unique(d['fem_cells']),np.arange(len(x)))
        # Map derivative must reproduce the identity, including curved elements.
        x=x.copy();x[:,1]+=.03*np.sin(x[:,0])*x[:,1]
        curved,_=build(x,c)
        gradient=np.einsum('eqia,eib->eqba',curved['fem_gradient'],x[curved['fem_cells']])
        np.testing.assert_allclose(gradient,np.broadcast_to(np.eye(2),gradient.shape),atol=2e-14)
        from examples.channel_nsch2d.data import geometry_data
        from examples.channel_nsch2d.geometry import nominal_parameters
        from examples.channel_nsch2d.online import validate_payload,INTEGER_INPUTS
        raw,_=geometry_data(nominal_parameters(c),c)
        actual={k:v.astype('int32' if k in INTEGER_INPUTS else c.train.compute_dtype) for k,v in raw.items()}
        validate_payload(actual,c)
        c.sample.Nx=7
        with self.assertRaises(ValueError):validate_config(c)

    def test_weak_rows_against_independent_element_integration(self):
        import jax
        import jax.numpy as jnp
        c,x,d,_=fixture();rng=np.random.default_rng(43)
        value=rng.normal(size=(len(x),5))*.2;rate=rng.normal(size=value.shape)*.1
        expected=oracle(value,rate,x,d,c)
        with jax.enable_x64(True):
            rows=make_weak_rows(c);device={k:jnp.asarray(v)for k,v in d.items()}
            actual=np.asarray(rows(jnp.asarray(value)[None],jnp.asarray(rate)[None],device))[0]
        np.testing.assert_allclose(actual,expected,atol=2e-14,rtol=1e-12)

    def test_outlet_transport_and_pressure_anchor(self):
        import jax
        import jax.numpy as jnp
        c,x,d,_=fixture();value=np.zeros((1,len(x),5));value[...,0]=1;value[...,2]=.2
        with jax.enable_x64(True):
            rows=make_weak_rows(c);device={k:jnp.asarray(v)for k,v in d.items()}
            actual=np.asarray(rows(jnp.asarray(value),jnp.zeros_like(jnp.asarray(value)),device))[0]
            np.testing.assert_allclose(actual*d['fem_active'],0.,atol=2e-14)
            value[...,4]=.7
            actual=np.asarray(rows(jnp.asarray(value),jnp.zeros_like(jnp.asarray(value)),device))[0]
        expected=np.zeros_like(actual)
        for cell in d['fem_outlet']:
            length=x[cell[1],1]-x[cell[0],1]
            np.add.at(expected[:,2],cell,-.7*length*np.array([1,1,4])/6)
        np.testing.assert_allclose(actual*d['fem_active'],expected*d['fem_active'],atol=2e-14)

    def test_reverse_derivative_and_full_network_loss(self):
        import equinox as eqx
        import jax
        import jax.numpy as jnp
        from examples.channel_nsch2d.data import geometry_data
        from examples.channel_nsch2d.geometry import nominal_parameters
        from examples.channel_nsch2d.network import build_network
        from examples.channel_nsch2d.losses import make_loss_fn,part_names
        c,x,d,_=fixture();rng=np.random.default_rng(9)
        with jax.enable_x64(True):
            rows=make_weak_rows(c);device={k:jnp.asarray(v)for k,v in d.items()}
            v=jnp.asarray(rng.normal(size=(2,len(x),5))*.1);direction=jnp.asarray(rng.normal(size=v.shape))
            f=jax.jit(lambda a:jnp.sum(rows(a,jnp.zeros_like(a),device)**2))
            g=jax.jit(jax.grad(f))(v);eps=1e-6
            np.testing.assert_allclose(jnp.sum(g*direction),(f(v+eps*direction)-f(v-eps*direction))/(2*eps),rtol=1e-8,atol=1e-8)
            c.arch.width=4;c.arch.heads=1;c.arch.condition_depth=1;c.arch.decoder_depth=1
            c.arch.spatial_features=2;c.arch.time_features=4;c.sample.Nt=2;c.train.microbatch.size=1
            c.sample.boundary_points=6;c.operator.condition_batch=1
            raw,_=geometry_data(nominal_parameters(c),c)
            data={k:jnp.asarray(v)[None]for k,v in raw.items()};times=jnp.array([.25,1.])
            net=build_network(c,jax.random.PRNGKey(8));loss=make_loss_fn(c)
            (v,parts),grad=eqx.filter_jit(eqx.filter_value_and_grad(loss,has_aux=True))(net,data,times)
            self.assertEqual(parts.shape,(4,));self.assertEqual(len(part_names(c)),4)
            self.assertTrue(np.isfinite(v));self.assertTrue(all(np.isfinite(a).all() for a in jax.tree.leaves(grad) if eqx.is_array(a)))
            evaluate=eqx.filter_jit(lambda m:loss(m,data,times)[0])
            eps=2e-5;idx=(0,0)
            plus=eqx.tree_at(lambda n:n.output.weight,net,net.output.weight.at[idx].add(eps))
            minus=eqx.tree_at(lambda n:n.output.weight,net,net.output.weight.at[idx].add(-eps))
            np.testing.assert_allclose(grad.output.weight[idx],(evaluate(plus)-evaluate(minus))/(2*eps),rtol=2e-6,atol=2e-7)


if __name__=='__main__':unittest.main()
