"""Independently check physical pressure jets and the frozen quadratic."""
import unittest
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from ml_collections import ConfigDict
from examples.channel_nsch2d.problem import validate_config
from examples.channel_nsch2d.configs.train_single_conservative import get_config
from examples.channel_nsch2d.network import build_network,encode,spatial_field_jets
from examples.channel_nsch2d.data import geometry_data
from examples.channel_nsch2d.pressure_head import feature_jets,quadratic_statistics


class ChannelPressureHead(unittest.TestCase):
    def test_chemical_pressure_coordinate_preserves_force_and_original_traction(self):
        c=get_config();c.sample.Nx=8;c.sample.Ny=8;c.sample.boundary_points=4
        c.arch.width=4;c.arch.heads=1;c.arch.condition_depth=1;c.arch.decoder_depth=1
        c.arch.spatial_features=2;c.arch.time_features=2
        shifted=ConfigDict(c.to_dict());shifted.pressure_coordinate=ConfigDict(dict(kind='chemical_shift'))
        validate_config(shifted)
        raw,_=geometry_data((.12,.08),c)
        data={k:jnp.asarray(v,dtype=v.dtype if np.issubdtype(v.dtype,np.integer) else jnp.float32) for k,v in raw.items()}
        model=build_network(c,jax.random.PRNGKey(52))
        points=jnp.array([[.8,.4],[2.,.45],[4.,.6]],dtype=jnp.float32)
        times=jnp.array([.25,2.],dtype=jnp.float32)
        @eqx.filter_jit
        def evaluate(net):
            e=encode(net,data)
            return (spatial_field_jets(net,e,points,times,data['amplitudes'],c),
                    spatial_field_jets(net,e,points,times,data['amplitudes'],shifted))
        original,changed=evaluate(model);v,x,y=[np.asarray(a) for a in original]
        q,qx,qy=[np.asarray(a) for a in changed]
        for a,b in zip(original,changed,strict=True):np.testing.assert_array_equal(a[...,:4],b[...,:4])
        np.testing.assert_allclose(q[...,4],v[...,4]-v[...,0]*v[...,1],atol=2e-6,rtol=2e-5)
        for old,d in ((x,qx),(y,qy)):
            expected=old[...,4]-old[...,0]*v[...,1]-v[...,0]*old[...,1]
            np.testing.assert_allclose(d[...,4],expected,atol=2e-6,rtol=2e-5)
            # Independent force forms: grad(p)+phi*grad(mu) and grad(Pi)-mu*grad(phi).
            np.testing.assert_allclose(d[...,4]+q[...,0]*d[...,1],
                old[...,4]-v[...,1]*old[...,0],atol=2e-6,rtol=2e-5)
        np.testing.assert_allclose(-q[...,4]+2*c.physical.viscosity*qx[...,2],
            -v[...,4]+v[...,0]*v[...,1]+2*c.physical.viscosity*x[...,2],atol=2e-6,rtol=2e-5)
        shifted.pressure_coordinate.kind='unknown'
        with self.assertRaisesRegex(ValueError,'Pressure coordinate'):validate_config(shifted)
        shifted.pressure_coordinate.kind='chemical_shift';del shifted.velocity
        with self.assertRaisesRegex(ValueError,'Pressure coordinate'):validate_config(shifted)

    def test_actual_pressure_value_gradient_and_frozen_other_fields(self):
        c=get_config();c.sample.Nx=8;c.sample.Ny=8;c.sample.boundary_points=4
        c.arch.width=4;c.arch.heads=1;c.arch.condition_depth=1;c.arch.decoder_depth=1
        c.arch.spatial_features=2;c.arch.time_features=2
        raw,_=geometry_data((.12,.08),c)
        data={k:jnp.asarray(v,dtype=v.dtype if np.issubdtype(v.dtype,np.integer) else jnp.float32) for k,v in raw.items()}
        model=build_network(c,jax.random.PRNGKey(42))
        x=jnp.array([[.8,.4],[1.5,.3],[3.,.6]],dtype=jnp.float32)
        times=jnp.array([.25,2.],dtype=jnp.float32)
        @eqx.filter_jit
        def check(net):
            e=encode(net,data)
            return feature_jets(net,e,x,times,c),spatial_field_jets(net,e,x,times,data['amplitudes'],c)
        basis,physical=check(model)
        w=jnp.concatenate((model.output.weight[3],model.output.bias[3,None]))
        for f,p in zip(basis,physical,strict=True):
            np.testing.assert_allclose(f@w,p[...,4],atol=2e-6,rtol=2e-5)
        delta=jnp.array([.1,-.2,.3,-.1,.05],dtype=jnp.float32)
        changed=eqx.tree_at(lambda n:(n.output.weight,n.output.bias),model,
            (model.output.weight.at[3].add(delta[:-1]),model.output.bias.at[3].add(delta[-1])))
        _,after=check(changed)
        for b,p,q in zip(basis,physical,after,strict=True):
            np.testing.assert_allclose(q[...,4]-p[...,4],b@delta,atol=2e-6,rtol=2e-5)
            np.testing.assert_array_equal(q[...,:4],p[...,:4])

    def test_quadratic_matches_direct_weighted_residual_and_gradient(self):
        rng=np.random.default_rng(47)
        B=rng.normal(size=(100,7));r=rng.normal(size=100);w=rng.uniform(.01,1,100)
        G,g,f=quadratic_statistics(B,r,w);d=rng.normal(size=7)
        np.testing.assert_allclose(f+2*g@d+d@G@d,np.sum(w*(r+B@d)**2),rtol=2e-15)
        np.testing.assert_allclose(2*(G@d+g),2*B.T@(w*(r+B@d)),atol=3e-13,rtol=2e-14)
        self.assertGreater(np.linalg.eigvalsh(G).min(),0.)
        with self.assertRaisesRegex(ValueError,'measure'):quadratic_statistics(B,r,-w)


if __name__=='__main__':unittest.main()
