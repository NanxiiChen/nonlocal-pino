"""Normalized streamfunction envelope: hard constraints and physical gradients."""
import json
from pathlib import Path
import tempfile
import unittest

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from examples.channel_nsch2d.causal import comparison_config
from examples.channel_nsch2d.data import geometry_data
from examples.channel_nsch2d.losses import make_loss_fn
from examples.channel_nsch2d.network import build_network,encode,fields,field_jets,potentials,decode,lifting_gates
from examples.channel_nsch2d.problem import walls,flow_rate,initial_field,validate_config
from tests.test_channel_time_jvp import config as original_config


def config():
    c=original_config();c.stream_lifting=dict(kind='normalized_inlet')
    return c


class ChannelStreamLiftingTests(unittest.TestCase):
    def test_constraints_flux_divergence_and_original_branch_parity(self):
        with jax.enable_x64(True):
            c=config();validate_config(c);raw,_=geometry_data((.12,.08),c)
            d={k:jnp.asarray(v)for k,v in raw.items()};net=build_network(c,jax.random.PRNGKey(8));e=encode(net,d)
            times=jnp.asarray([0.,.6,2.]);boundary=d['boundary'];x=boundary[:,:2]
            predict=eqx.filter_jit(lambda points:fields(net,e,points,times,d['amplitudes'],c))
            value=np.asarray(predict(x));wall=np.asarray(boundary[:,4])==0;inlet=np.asarray(boundary[:,4])==1
            np.testing.assert_allclose(value[:,wall,2:4],0.,rtol=0,atol=2e-14)
            np.testing.assert_array_equal(value[0,:,0],initial_field(np.asarray(x),c,xp=np))
            np.testing.assert_array_equal(value[0,:,2:4],0.)
            expected=6*np.asarray(flow_rate(times,c,xp=jnp))[:,None]*np.asarray(x[inlet,1])*(1-np.asarray(x[inlet,1]))
            np.testing.assert_allclose(value[:,inlet,2],expected,rtol=0,atol=2e-14)
            np.testing.assert_allclose(value[:,inlet,3],0.,rtol=0,atol=2e-14)
            # The neural velocity is not polynomial in eta. Independently
            # checked 40/80/160-point convergence selects160 without changing
            # the2e-11 flux gate (errors2.47e-6/5.12e-10/2.13e-13).
            order=160;roots,weights=np.polynomial.legendre.leggauss(order);eta=(roots+1)/2
            section=np.array([0.,.001,.03,.1,.8,2.,3.9,4.]);xx,ee=np.meshgrid(section,eta,indexing='ij')
            b,t=walls(xx,(.12,.08),c,xp=np);points=jnp.asarray(np.stack((xx,b+ee*(t-b)),axis=-1).reshape(-1,2))
            velocity=np.asarray(predict(points))[...,2].reshape(3,len(section),order)
            flux=np.sum(velocity*(t-b)[None]*weights[None,None,:]/2,axis=-1)
            np.testing.assert_allclose(flux,np.broadcast_to(np.asarray(flow_rate(times,c,xp=jnp))[:,None],flux.shape),rtol=0,atol=2e-11)
            probe=jnp.asarray([[.001,.3],[.03,.7],[.8,.4],[2.,.6],[3.9,.5]])
            _,_,dx,dy=eqx.filter_jit(lambda x:field_jets(net,e,x,times,d['amplitudes'],c))(probe)
            np.testing.assert_allclose(dx[...,2]+dy[...,3],0.,rtol=0,atol=3e-12)
            old=original_config();p=probe;bottom,top=walls(p[:,0],d['amplitudes'],old,xp=jnp);eta=(p[:,1]-bottom)/(top-bottom)
            unlifted=decode(net,net.prepare_encoded(e,p),times,old);_,time=lifting_gates(p,times,old)
            expected=flow_rate(times,old,xp=jnp)[:,None]*(3*eta**2-2*eta**3)[None,:]
            expected+=old.loss.stream_scale*time[:,None]*(16*(eta*(1-eta))**2*(p[:,0]/old.geometry.length)**2)[None,:]*unlifted[...,2]
            actual=potentials(net,e,p,times,d['amplitudes'],old)[...,2]
            np.testing.assert_array_equal(actual,expected)
            self.assertGreater(float(jnp.max(jnp.abs(potentials(net,e,p,times,d['amplitudes'],c)[...,2]-actual))),1e-5)

    def test_full_physical_residual_parameter_gradient(self):
        with jax.enable_x64(True):
            c=config();raw=[geometry_data(a,c)[0]for a in ((.12,.08),(.06,.13))]
            batch={k:jnp.asarray(np.stack([d[k]for d in raw]))for k in raw[0]}
            net=build_network(c,jax.random.PRNGKey(9));times=jnp.asarray([.037,.22,.71,2.37])
            loss=make_loss_fn(c)
            (total,parts),gradient=eqx.filter_jit(eqx.filter_value_and_grad(loss,has_aux=True))(net,batch,times)
            self.assertTrue(np.isfinite(total));self.assertEqual(parts.shape,(9,))
            self.assertTrue(all(np.isfinite(a).all()for a in jax.tree.leaves(gradient)if eqx.is_array(a)))
            evaluate=eqx.filter_jit(lambda n:loss(n,batch,times)[0]);h=1e-5
            for index in ((0,0),(2,0)):
                plus=eqx.tree_at(lambda n:n.output.weight,net,net.output.weight.at[index].add(h))
                minus=eqx.tree_at(lambda n:n.output.weight,net,net.output.weight.at[index].add(-h))
                numeric=float((evaluate(plus)-evaluate(minus))/(2*h));self.assertGreater(abs(numeric),1e-8)
                np.testing.assert_allclose(gradient.output.weight[index],numeric,rtol=3e-5,atol=3e-6)

    def test_strict_schema_and_matched_jvp_comparison(self):
        c=config();validate_config(c)
        for field in ('velocity','lifting'):
            c=config();del c[field]
            with self.assertRaisesRegex(ValueError,'Stream lifting'):validate_config(c)
        c=config();c.stream_lifting.kind='linear'
        with self.assertRaisesRegex(ValueError,'Stream lifting'):validate_config(c)
        c=config();c.stream_lifting.scale=.1
        with self.assertRaisesRegex(ValueError,'Stream lifting'):validate_config(c)
        with tempfile.TemporaryDirectory()as directory:
            root=Path(directory);saved=original_config();current=config()
            for name in ('sampling.npz','prediction.npz','known_inputs.npz','known_inputs_final.npz','model_initial.eqx','optimizer_initial.eqx'):
                (root/name).touch()
            (root/'config.json').write_text(saved.to_json())
            (root/'summary.json').write_text(json.dumps(dict(status='completed',iterations=saved.train.iterations)))
            (root/'validation.json').write_text(json.dumps(dict(status='replay_and_hard_constraints_passed')))
            self.assertEqual(comparison_config(current,root),json.loads(saved.to_json()))
            current.train.lr_init*=2
            with self.assertRaisesRegex(ValueError,'only one explicit'):comparison_config(current,root)
            current=config();del saved.time_derivative
            (root/'config.json').write_text(saved.to_json())
            with self.assertRaisesRegex(ValueError,'completed original-envelope adaptiveJVP'):comparison_config(current,root)


if __name__=='__main__':unittest.main()
