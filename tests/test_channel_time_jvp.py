"""Physical timeJVP, temporal consistency, full gradients and fresh-time training."""
import contextlib
from concurrent.futures import ThreadPoolExecutor
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from examples.channel_nsch2d.configs.train_operator_width128_gmls_time40_adaptive_jvp_online import get_config
from examples.channel_nsch2d.causal import make_causal_fn,audit_trace
from examples.channel_nsch2d.data import geometry_data
from examples.channel_nsch2d.losses import make_loss_fn
from examples.channel_nsch2d.network import build_network,encode,fields,time_jets
from examples.channel_nsch2d.problem import validate_config,identity
from examples.channel_nsch2d.temporal import sample_continuous,backward_rate
from tests.test_channel_gmls_first import small


def config():
    c=small();preset=get_config();del c.time_discretization
    c.time_derivative=preset.time_derivative;c.causal=preset.causal
    c.train.microbatch=preset.train.microbatch;c.sample.Nt=40
    c.operator.condition_batch=2;c.train.iterations=5000;c.train.soap_max_precond_dim=32
    return c


class ChannelTimeJVPTests(unittest.TestCase):
    def test_physical_time_jvp_independent_jacobian_and_bdf_convergence(self):
        with jax.enable_x64(True):
            c=config();validate_config(c);raw,_=geometry_data((.12,.08),c)
            d={k:jnp.asarray(v)for k,v in raw.items()};net=build_network(c,jax.random.PRNGKey(8))
            encoded=encode(net,d);points=d['points'][::45][:3];times=jnp.asarray([.13,.71,2.37])
            physical=eqx.filter_jit(lambda t:fields(net,encoded,points,t,d['amplitudes'],c))
            value,rate=eqx.filter_jit(lambda n,e,x,t,a:time_jets(n,e,x,t,a,c))(
                net,encoded,points,times,d['amplitudes'])
            np.testing.assert_allclose(value,physical(times),rtol=2e-12,atol=2e-12)
            jac=eqx.filter_jit(jax.jacfwd(physical))(times)
            independent=jnp.stack([jac[i,:,:,i]for i in range(len(times))])
            np.testing.assert_allclose(rate,independent,rtol=2e-11,atol=2e-12)
            for i in range(len(times)):jac=jac.at[i,:,:,i].set(0.)
            np.testing.assert_array_equal(jac,jnp.zeros_like(jac))
            h=1e-5;fd=(physical(times+h)-physical(times-h))/(2*h)
            np.testing.assert_allclose(rate,fd,rtol=2e-6,atol=2e-8)
            errors=[]
            for h in (.02,.01,.005):
                bdf=backward_rate(value,physical(times-h),physical(times-2*h),times,h)
                errors.append(float(jnp.linalg.norm(bdf-rate)))
            self.assertTrue(all(b/a<.4 for a,b in zip(errors[:-1],errors[1:],strict=True)),errors)

    def test_time_jvp_complete_parameter_gradient_and40time_accumulation(self):
        with jax.enable_x64(True):
            c=config();raw=[geometry_data(a,c)[0]for a in ((.12,.08),(.06,.13))]
            batch={k:jnp.asarray(np.stack([d[k]for d in raw]))for k in raw[0]}
            model=build_network(c,jax.random.PRNGKey(9));times=jax.jit(lambda k:sample_continuous(k,c))(jax.random.PRNGKey(5)).astype(jnp.float64)
            loss=make_loss_fn(c)
            (total,parts),gradient=eqx.filter_jit(eqx.filter_value_and_grad(loss,has_aux=True))(model,batch,times)
            accumulated=eqx.filter_jit(make_causal_fn(c))
            value,vector,actual,_=accumulated(model,batch,times,jnp.asarray(0.))
            np.testing.assert_allclose(value,total,rtol=2e-12,atol=2e-12)
            np.testing.assert_allclose(vector,parts,rtol=2e-12,atol=2e-12)
            for a,b in zip(jax.tree.leaves(actual),jax.tree.leaves(gradient),strict=True):
                np.testing.assert_allclose(a,b,rtol=2e-10,atol=2e-11)
            evaluate=eqx.filter_jit(lambda n:loss(n,batch,times)[0]);h=1e-5;index=(0,0)
            plus=eqx.tree_at(lambda n:n.output.weight,model,model.output.weight.at[index].add(h))
            minus=eqx.tree_at(lambda n:n.output.weight,model,model.output.weight.at[index].add(-h))
            numeric=float((evaluate(plus)-evaluate(minus))/(2*h))
            self.assertGreater(abs(numeric),1e-8)
            np.testing.assert_allclose(gradient.output.weight[index],numeric,rtol=2e-5,atol=2e-6)

    def test_explicit_time_mode_and_fresh_continuous_sampler(self):
        c=get_config();validate_config(c)
        keys=jax.random.split(jax.random.PRNGKey(4),4)
        draws=np.asarray(jax.jit(jax.vmap(lambda k:sample_continuous(k,c)))(keys))
        self.assertEqual(draws.shape,(4,40));self.assertFalse(np.array_equal(draws[0],draws[1]))
        self.assertTrue(np.all(np.diff(draws,axis=-1)>0))
        strata=np.arange(40)*c.physical.terminal_time/40
        self.assertTrue(np.all((draws>=strata)&(draws<strata+c.physical.terminal_time/40)))
        self.assertGreater(np.max(np.abs(draws/.025-np.rint(draws/.025))),.1)
        for key,value in [('kind','bdf2'),('sampling','fixed')]:
            c=get_config();setattr(c.time_derivative,key,value)
            with self.assertRaisesRegex(ValueError,'Explicit time derivative'):validate_config(c)
        c=get_config();c.time_discretization=dict(kind='bdf2',dt=.025)
        with self.assertRaisesRegex(ValueError,'Explicit time derivative'):validate_config(c)
        c=get_config();del c.time_derivative
        with self.assertRaisesRegex(ValueError,'GMLS-first'):validate_config(c)

    def test_actual_continuous_online_canonical_training_and_feedback_replay(self):
        from examples.channel_nsch2d import train as entry
        from examples.channel_nsch2d.cohort import prepare
        from examples.channel_nsch2d.online import audit_training_inputs
        c=config();c.train.iterations=3;c.operator.online.refresh_every=1
        c.operator.online.refresh_size=2;c.log.eval_every=3;c.log.checkpoint_every=3
        with tempfile.TemporaryDirectory()as directory,jax.enable_x64(True):
            root=Path(directory);c.reference.root=str(root);c.log.log_dir=str(root/'runs');c.log.run_name='time-jvp'
            c.operator.validation_manifest=str(root/'reserved/manifest.json');prepare(c,c.operator.validation_manifest,4,(96,40))
            target=np.ones((13,3,5));target[...,0]=.5;reference=root/'reference.npz'
            np.savez(reference,points=np.array([[.2,.3],[.8,.5],[2.,.5]]),time=np.linspace(0,6,13),amplitudes=np.array([.12,.08]),fields=target)
            (root/'reference-validation.json').write_text(json.dumps(dict(status='passed',reference=str(reference),identity=identity(c))))
            def executor(**kwargs):return ThreadPoolExecutor(max_workers=kwargs['max_workers'])
            with patch('examples.channel_nsch2d.online.ProcessPoolExecutor',executor),contextlib.redirect_stdout(io.StringIO()):entry.train(c)
            run=Path(c.log.log_dir)/c.log.run_name;trace=audit_trace(run,c);pool=audit_training_inputs(run,c)
            self.assertEqual(trace['feedback']['updates'],3);self.assertEqual(pool['refresh_count'],2)
            with np.load(run/'sampling.npz',allow_pickle=False)as f:
                self.assertNotIn('intervals',f.files);stored=f['times']
                _,key=jax.random.split(jax.random.PRNGKey(c.train.seed))
                def expected(i):
                    _,kt=jax.random.split(jax.random.fold_in(key,i));return sample_continuous(kt,c)
                np.testing.assert_array_equal(stored,np.asarray(jax.jit(jax.vmap(expected))(jnp.arange(3,dtype=jnp.uint32))))
                self.assertFalse(np.array_equal(stored[0],stored[1]))
            summary=json.loads((run/'summary.json').read_text())
            self.assertEqual(summary['time_derivative']['kind'],'jvp');self.assertNotIn('time_discretization',summary)
            self.assertIn('GMLS PDE/BC first',summary['method']);self.assertIn('time JVP',summary['method'])
            self.assertNotEqual((run/'model_initial.eqx').read_bytes(),(run/'model_final.eqx').read_bytes())
            from examples.channel_nsch2d.diagnose_time import diagnose
            with contextlib.redirect_stdout(io.StringIO()):
                diagnostic=diagnose(run,root/'continuous-audit',8,731)
            self.assertEqual(len(diagnostic['cases']),2)
            self.assertEqual(diagnostic['time_samples'],8);self.assertEqual(diagnostic['reference_labels_used'],0)
            for row in diagnostic['cases']:
                self.assertEqual(row['domain_nodes'],(c.sample.Nx+1)*(c.sample.Ny+1))
                with np.load(row['source'],allow_pickle=False)as f:
                    self.assertEqual(f['parts'].shape,(8,9));self.assertTrue(np.all(np.diff(f['times'])>0))


if __name__=='__main__':unittest.main()
