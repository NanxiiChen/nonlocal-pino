"""Projected nodal chemical fields and explicit continuous-time queries."""
import unittest

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from examples.channel_nsch2d.data import geometry_data
from examples.channel_nsch2d.network import build_network,encode,fields
from examples.channel_nsch2d.problem import walls,material
from examples.channel_nsch2d.projected_fields import query_rule,make_domain_fn,make_domain_time_fn,make_query_fn
from examples.channel_nsch2d.chemical_projection import project_rate,scalar_laplacian
from tests.test_channel_time_jvp import config as base_config


def config():
    c=base_config();c.chemical_constraint=dict(kind='interior_gmls_projection',boundary='decoder_trace',query='mapped_bilinear')
    return c


class ChannelProjectedFieldTests(unittest.TestCase):
    def test_query_rule_exact_nodes_partition_and_nonzero_mapped_linear(self):
        c=config();raw,_=geometry_data((.12,.08),c);source=raw['points']
        rule=query_rule(raw,source,c)
        np.testing.assert_array_equal(rule['indices'][:,0],np.arange(len(source)))
        np.testing.assert_array_equal(rule['weights'][:,0],1.);self.assertFalse(rule['boundary_trace'].any())
        x=np.array([.13,.8,1.3,2.2,3.89]);eta=np.array([.15,.72,.36,.88,.44]);bottom,top=walls(x,raw['amplitudes'],c,xp=np)
        points=np.stack((x,bottom+eta*(top-bottom)),axis=-1);rule=query_rule(raw,points,c)
        xx,ee=np.meshgrid(np.linspace(0,4,c.sample.Nx+1),np.linspace(0,1,c.sample.Ny+1),indexing='ij')
        value=(.3+.2*xx-.4*ee).ravel();interpolated=np.sum(value[rule['indices']]*rule['weights'],axis=-1)
        np.testing.assert_allclose(interpolated,.3+.2*x-.4*eta,rtol=0,atol=2e-14)
        np.testing.assert_allclose(rule['weights'].sum(-1),1.,rtol=0,atol=2e-14)
        with self.assertRaisesRegex(ValueError,'outside'):query_rule(raw,np.array([[.4,-.1]]),c)
        bad=dict(raw);bad['points']=raw['points'][::-1]
        with self.assertRaisesRegex(ValueError,'ordering'):query_rule(bad,points,c)

    def test_physical_domain_time_chain_and_trace(self):
        with jax.enable_x64(True):
            c=config();raw,_=geometry_data((.12,.08),c);data={k:jnp.asarray(v)for k,v in raw.items()}
            net=build_network(c,jax.random.PRNGKey(19));e=encode(net,data);times=jnp.asarray([.13,.75,2.3])
            value,rate=eqx.filter_jit(make_domain_time_fn(c))(net,e,data,times)
            old=base_config();primal=fields(net,e,data['points'],times,data['amplitudes'],old)
            _,raw_rate=jax.jvp(lambda t:fields(net,e,data['points'],t,data['amplitudes'],old),(times,),(jnp.ones_like(times),))
            expected=project_rate(primal,raw_rate,data,c,xp=jnp)
            np.testing.assert_allclose(rate,expected,rtol=2e-11,atol=2e-11)
            inside=raw['interior']>0;A,kappa=material(c)
            chemical=value[...,1]-A*(value[...,0]**3-value[...,0])+kappa*scalar_laplacian(value[...,0],data,xp=jnp)
            np.testing.assert_allclose(np.asarray(chemical)[:,inside],0.,rtol=0,atol=2e-12)
            np.testing.assert_allclose(np.asarray(value)[:,~inside,1],np.asarray(primal)[:,~inside,1],rtol=2e-12,atol=2e-12)

    def test_query_native_identity_boundary_trace_and_parameter_finite_difference(self):
        with jax.enable_x64(True):
            c=config();raw,_=geometry_data((.12,.08),c);data={k:jnp.asarray(v)for k,v in raw.items()}
            net=build_network(c,jax.random.PRNGKey(20));times=jnp.asarray([.2,1.1])
            ids=np.array([20,45,70]);points=np.concatenate((raw['points'][ids],raw['boundary'][:4,:2]))
            host_rule=query_rule(raw,points,c);rule={k:jnp.asarray(v)for k,v in host_rule.items()}
            fn=make_query_fn(c);domain=make_domain_fn(c);points=jnp.asarray(points)
            @eqx.filter_jit
            def predict(model):return fn(model,encode(model,data),data,points,times,rule)
            value=predict(net);e=encode(net,data);native=domain(net,e,data,times)
            np.testing.assert_allclose(value[:,:3,1],native[:,ids,1],rtol=2e-12,atol=2e-12)
            rawc=base_config();trace=fields(net,e,points,times,data['amplitudes'],rawc)
            mask=host_rule['boundary_trace'];self.assertTrue(mask.any())
            np.testing.assert_allclose(np.asarray(value)[:,mask,1],np.asarray(trace)[:,mask,1],rtol=2e-12,atol=2e-12)
            objective=eqx.filter_jit(lambda model:jnp.mean(predict(model)**2))
            _,gradient=eqx.filter_jit(eqx.filter_value_and_grad(objective))(net)
            h=1e-5
            for index in ((0,0),(1,0)):
                plus=eqx.tree_at(lambda n:n.output.weight,net,net.output.weight.at[index].add(h))
                minus=eqx.tree_at(lambda n:n.output.weight,net,net.output.weight.at[index].add(-h))
                numeric=float((objective(plus)-objective(minus))/(2*h));self.assertGreater(abs(numeric),1e-8)
                np.testing.assert_allclose(gradient.output.weight[index],numeric,rtol=3e-5,atol=3e-6)


if __name__=='__main__':unittest.main()
