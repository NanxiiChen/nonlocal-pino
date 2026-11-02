"""Actual coupled gradients, cross-block prefixes and canonical online updates."""
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

from examples.channel_nsch2d.causal import make_causal_fn,epsilon_at,audit_trace,make_epsilon_update
from examples.channel_nsch2d.configs.train_operator_width128_gmls_time40_causal_online import get_config
from examples.channel_nsch2d.configs.train_operator_width128_gmls_time40_adaptive_causal_online import get_config as adaptive_config
from examples.channel_nsch2d.data import geometry_data
from examples.channel_nsch2d.losses import make_loss_fn,make_time_parts_fn
from examples.channel_nsch2d.network import build_network
from examples.channel_nsch2d.problem import validate_config,identity
from examples.channel_nsch2d.temporal import sample_intervals
from tests.test_channel_gmls_first import small


def config():
    c=small();c.sample.Nt=40;c.operator.condition_batch=2
    p=get_config();c.train.microbatch=p.train.microbatch;c.causal=p.causal
    c.train.iterations=5000;c.train.soap_max_precond_dim=32
    return c


class ChannelCausalTests(unittest.TestCase):
    def test_global40_prefix_numpy_weights_frozen_gradient_and_zero_epsilon(self):
        with jax.enable_x64(True):
            c=config();raw=[geometry_data(a,c)[0] for a in ((.12,.08),(.06,.13))]
            batch={k:jnp.asarray(np.stack([d[k] for d in raw])) for k in raw[0]}
            model=build_network(c,jax.random.PRNGKey(9))
            _,times=sample_intervals(jax.random.PRNGKey(8),c)
            time_parts=make_time_parts_fn(c)
            fn=eqx.filter_jit(make_causal_fn(c))
            strength=1e-4
            total,parts,gradient,aux=fn(model,batch,times,jnp.asarray(strength))
            primal=eqx.filter_jit(time_parts)(model,batch,times)
            seg=np.asarray(primal).sum(-1)
            w=np.exp(-strength*np.c_[np.zeros(2),np.cumsum(seg[:,:-1],axis=-1)])
            np.testing.assert_allclose(aux['segments'],seg,rtol=1e-12,atol=1e-12)
            np.testing.assert_allclose(aux['weights'],w,rtol=1e-12,atol=1e-12)
            np.testing.assert_allclose(parts,(np.asarray(primal)*w[:,:,None]).mean((0,1)),rtol=1e-12,atol=1e-12)
            np.testing.assert_allclose(total,np.mean(w*seg),rtol=1e-12,atol=1e-12)
            self.assertTrue(np.all(w[:,8]<1.))
            self.assertGreater(np.max(np.abs(w[0]-w[1])),1e-8)
            frozen=jnp.asarray(w)
            expected=eqx.filter_jit(eqx.filter_grad(lambda m:jnp.mean(jnp.sum(time_parts(m,batch,times),axis=-1)*frozen)))(model)
            for a,b in zip(jax.tree.leaves(gradient),jax.tree.leaves(expected),strict=True):
                np.testing.assert_allclose(a,b,rtol=2e-10,atol=2e-12)
            def weight_sum(shift):
                changed=eqx.tree_at(lambda m:m.output.bias,model,model.output.bias+jnp.array([shift,0.,0.,0.]))
                return fn(changed,batch,times,jnp.asarray(strength))[3]['weights'].sum()
            self.assertEqual(float(jax.grad(weight_sum)(0.)),0.)
            (plain,vector),g0=eqx.filter_jit(eqx.filter_value_and_grad(make_loss_fn(c),has_aux=True))(model,batch,times)
            zero,zparts,zgrad,_=fn(model,batch,times,jnp.asarray(0.))
            np.testing.assert_allclose(zero,plain,rtol=1e-12,atol=1e-12)
            np.testing.assert_allclose(zparts,vector,rtol=1e-12,atol=1e-12)
            for a,b in zip(jax.tree.leaves(zgrad),jax.tree.leaves(g0),strict=True):
                np.testing.assert_allclose(a,b,rtol=2e-10,atol=2e-12)
            with self.assertRaises(Exception) as caught:fn(model,batch,times[::-1],jnp.asarray(strength))
            self.assertIn('strictly increasing',str(caught.exception))

    def test_schedule_boundaries_and_invalid_configs(self):
        c=get_config();validate_config(c)
        self.assertEqual([epsilon_at(i,c) for i in (0,999,1000,1999,2000,3000,4000,4999)],
                         [.01,.01,.1,.1,1.,10.,100.,100.])
        for eps in ((0.,1.),(float('nan'),1.),(1.,.1)):
            c=get_config();c.causal.epsilon_values=eps
            with self.assertRaisesRegex(ValueError,'epsilon_values'):validate_config(c)
        c=get_config();c.causal.steps_per_epsilon=999
        with self.assertRaisesRegex(ValueError,'budget'):validate_config(c)
        c=get_config();c.causal.boundary='unweighted'
        with self.assertRaisesRegex(ValueError,'same_time_weights'):validate_config(c)
        c=get_config();del c.train.microbatch
        with self.assertRaisesRegex(ValueError,'cold GMLS'):validate_config(c)

    def test_canonical3_update_online_train_and_independent_trace(self):
        self.canonical_trace(adaptive=False)

    def test_adaptive_canonical3_updates_and_feedback_tamper_detection(self):
        self.canonical_trace(adaptive=True)

    def test_existing_feedback_increase_hold_backoff_geometry_disagreement_and_cap(self):
        c=adaptive_config();validate_config(c);fn=eqx.filter_jit(make_epsilon_update(c))
        all_one=jnp.ones((2,40));hold=all_one.at[:,1:].set(.8);poor=all_one.at[:,1:].set(0.)
        cases=((all_one,2.,2.2),(hold,2.,2.),(poor,2.,2./1.1),
               (all_one.at[1].set(poor[1]),2.,2./1.1),
               (all_one.at[1].set(hold[1]),2.,2.),(all_one,99.,100.))
        from nonlocal_pinn.training.causal import CausalWeightor
        cfg=c.causal.adaptation;shared=CausalWeightor(n_chunks=None)
        for w,epsilon,expected in cases:
            eps=jnp.asarray(epsilon)
            actual=fn(w,eps)
            np.testing.assert_allclose(actual,expected,rtol=2e-6)
            proposals=[shared.update_eps(v,eps,cfg.step_size,cfg.max_eps,
                cfg.max_last_weight,cfg.min_mean_weight) for v in w]
            np.testing.assert_array_equal(actual,jnp.min(jnp.stack(proposals)))
        np.testing.assert_array_equal(jax.grad(lambda w:fn(w,jnp.asarray(2.)))(hold),jnp.zeros_like(hold))
        for field,value in [('min_mean_weight',.01),('max_last_weight',1.),('step_size',1.),
                            ('eps_init',0.),('max_eps',float('nan')),('step_size',True),
                            ('geometry_reduction','mean')]:
            c=adaptive_config();setattr(c.causal.adaptation,field,value)
            with self.assertRaisesRegex(ValueError,'adaptation'):validate_config(c)
        c=adaptive_config();c.causal.epsilon_values=(1.,2.)
        with self.assertRaisesRegex(ValueError,'Causal requires explicit'):validate_config(c)

    def canonical_trace(self,*,adaptive):
        from examples.channel_nsch2d import train as entry
        from examples.channel_nsch2d.cohort import prepare
        from examples.channel_nsch2d.online import audit_training_inputs
        c=config();c.train.iterations=3;c.causal.steps_per_epsilon=1
        c.causal.epsilon_values=(.01,.1,1.)
        if adaptive:c.causal=adaptive_config().causal
        c.operator.online.refresh_every=1;c.operator.online.refresh_size=2
        c.log.eval_every=3;c.log.checkpoint_every=3
        with tempfile.TemporaryDirectory() as directory,jax.enable_x64(True):
            root=Path(directory);c.reference.root=str(root);c.log.log_dir=str(root/'runs');c.log.run_name='causal'
            c.operator.validation_manifest=str(root/'reserved/manifest.json')
            prepare(c,c.operator.validation_manifest,4,(96,40))
            target=np.ones((13,3,5));target[...,0]=.5
            reference=root/'reference.npz'
            np.savez(reference,points=np.array([[.2,.3],[.8,.5],[2.,.5]]),
                time=np.linspace(0,6,13),amplitudes=np.array([.12,.08]),fields=target)
            (root/'reference-validation.json').write_text(json.dumps(dict(status='passed',reference=str(reference),identity=identity(c))))
            def executor(**kwargs):return ThreadPoolExecutor(max_workers=kwargs['max_workers'])
            with patch('examples.channel_nsch2d.online.ProcessPoolExecutor',executor),contextlib.redirect_stdout(io.StringIO()):
                entry.train(c)
            run=Path(c.log.log_dir)/'causal';trace=audit_trace(run,c);pool=audit_training_inputs(run,c)
            self.assertEqual(trace['records'],2 if adaptive else 3);self.assertEqual(pool['refresh_count'],2)
            self.assertNotEqual((run/'model_initial.eqx').read_bytes(),(run/'model_final.eqx').read_bytes())
            summary=json.loads((run/'summary.json').read_text())
            if adaptive:
                self.assertEqual(trace['feedback']['updates'],3)
                path=run/'causal-controller.jsonl';original=path.read_text()
                states=[json.loads(s) for s in original.splitlines()]
                self.assertEqual(states[0]['epsilon'],c.causal.adaptation.eps_init)
                self.assertEqual(states[-1]['epsilon'],summary['causal']['final_epsilon'])
                self.assertEqual(states[-1]['next_epsilon'],summary['causal']['next_epsilon'])
                states[0]['next_epsilon']*=1.1
                path.write_text('\n'.join(json.dumps(s) for s in states)+'\n')
                with self.assertRaisesRegex(ValueError,'state transition'):audit_trace(run,c)
                path.write_text('\n'.join(original.splitlines()[:-1])+'\n')
                with self.assertRaisesRegex(ValueError,'Incomplete actual causal feedback'):audit_trace(run,c)
                path.write_text(original);audit_trace(run,c)
            else:self.assertEqual(summary['causal']['final_epsilon'],1.)
            self.assertEqual(summary['microbatch']['optimizer_updates_per_batch'],1)
            self.assertTrue((run/'source/src/nonlocal_pinn/training/causal.py').is_file())


if __name__=='__main__':unittest.main()
