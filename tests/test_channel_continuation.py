"""Complete SOAP import, strict numerical guards and explicit mesh refinement."""
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from ml_collections import ConfigDict
from examples.channel_nsch2d.configs.train_single_conservative import get_config
from examples.channel_nsch2d.continuation import continuation_spec,restore_state
from nonlocal_pinn.training.optimizers import make_optimizer


def parent_fixture(path):
    c=get_config();c.train.seed=42;c.train.iterations=3
    model=eqx.nn.Linear(3,2,key=jax.random.PRNGKey(8));optimizer=make_optimizer(c)
    state=optimizer.init(eqx.filter(model,eqx.is_array))
    loss=lambda n:jnp.sum(n.weight**2)+jnp.sum(n.bias**2)
    for _ in range(3):
        updates,state=optimizer.update(eqx.filter_grad(loss)(model),state,model)
        model=eqx.apply_updates(model,updates)
    (path/'config.json').write_text(c.to_json())
    (path/'summary.json').write_text(json.dumps(dict(status='completed',iterations=3,metrics=dict(step=3),wall_seconds=1.2)))
    (path/'validation.json').write_text(json.dumps(dict(status='replay_and_hard_constraints_passed')))
    np.savez(path/'prediction.npz',fields=np.zeros((2,1,5)))
    eqx.tree_serialise_leaves(path/'model_final.eqx',model);eqx.tree_serialise_leaves(path/'optimizer_final.eqx',state)
    return c,model,optimizer,state,loss


class ChannelContinuation(unittest.TestCase):
    def test_time_refinement_preserves_outputs_state_and_all_other_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            c,model,optimizer,state,_=parent_fixture(Path(directory));c.train.iterations=8
            c.train.continuation=ConfigDict(dict(parent_run=directory,kind='time_refine'))
            c.time_discretization.dt=.025
            spec=continuation_spec(c)
            self.assertEqual(spec['time_refinement'],dict(before=dict(kind='bdf2',dt=.05),after=dict(kind='bdf2',dt=.025)))
            restored,optim,start=restore_state(model,optimizer,spec)
            self.assertEqual(start,3)
            for a,b in zip(jax.tree.leaves((model,state)),jax.tree.leaves((restored,optim)),strict=True):
                if eqx.is_array(a):np.testing.assert_array_equal(a,b)
            for dt in (.05,.1,.03,0.,float('nan'),True):
                c.time_discretization.dt=dt
                with self.assertRaisesRegex(ValueError,'subdivision'):continuation_spec(c)
            c.time_discretization.dt=.025;c.nl.degree=4
            with self.assertRaisesRegex(ValueError,'configuration'):continuation_spec(c)
            c.nl.degree=3;c.loss.chemical_scale=2.
            with self.assertRaisesRegex(ValueError,'configuration'):continuation_spec(c)

    def test_operator_refinement_cannot_change_condition_limit_or_physical_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            c,_,_,_,_=parent_fixture(Path(directory));c.train.iterations=8
            c.train.continuation=ConfigDict(dict(parent_run=directory,kind='operator'))
            c.nl.degree=4;c.nl.neighbors=48;c.nl.weight_exponent=12.
            spec=continuation_spec(c)
            self.assertEqual(spec['operator_refinement']['before']['degree'],3)
            self.assertEqual(spec['operator_refinement']['after']['degree'],4)
            c.nl.condition_limit*=10
            with self.assertRaisesRegex(ValueError,'configuration'):continuation_spec(c)
            c.nl.condition_limit/=10;c.loss.chemical_scale=1
            with self.assertRaisesRegex(ValueError,'configuration'):continuation_spec(c)

    def test_canonical_entry_matches_uninterrupted_state_and_global_draws(self):
        import contextlib
        import io
        from examples.channel_nsch2d import train as entry
        from examples.channel_nsch2d.problem import identity
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);c=get_config();c.train.seed=42;c.train.iterations=3
            c.reference.root=str(root);c.log.log_dir=str(root/'runs');c.log.run_name='parent'
            c.log.eval_every=1;c.log.checkpoint_every=1
            reference=root/'reference.npz'
            target=np.ones((13,3,5));target[...,0]=.5
            np.savez(reference,points=np.array([[.2,.3],[.8,.5],[2.,.5]]),
                time=np.linspace(0,6,13),amplitudes=np.array([.12,.08]),fields=target)
            (root/'reference-validation.json').write_text(json.dumps(dict(status='passed',reference=str(reference),identity=identity(c))))
            def payload(a,cc):return dict(amplitudes=np.asarray(a)),{}
            def bank(cc):return dict(amplitudes=np.array([[.12,.08]])),[]
            def model(cc,key):return eqx.nn.Linear(3,2,key=key)
            def physical(n,e,x,t,a,cc):return jnp.full((len(t),len(x),5),jnp.sum(n.weight))
            def loss(cc):
                def fn(n,batch,times):
                    total=jnp.sum(n.weight**2)+jnp.sum(n.bias**2)
                    return total,jnp.full(9,total/9)
                return fn
            # Mock only physical payload/forward/loss costs. The canonical
            # optimizer, serialization, import, loop and global sampler run.
            with patch.object(entry,'build_bank',bank),patch.object(entry,'geometry_data',payload),\
                 patch.object(entry,'build_network',model),patch.object(entry,'encode',lambda n,d:None),\
                 patch.object(entry,'fields',physical),patch.object(entry,'make_loss_fn',loss),contextlib.redirect_stdout(io.StringIO()):
                entry.train(c);parent=Path(c.log.log_dir)/'parent'
                (parent/'validation.json').write_text(json.dumps(dict(status='replay_and_hard_constraints_passed')))
                child=ConfigDict(c.to_dict());child.train.iterations=6;child.log.run_name='child'
                child.train.continuation=ConfigDict(dict(parent_run=str(parent),kind='strict'))
                entry.train(child)
                full=ConfigDict(c.to_dict());full.train.iterations=6;full.log.run_name='full';entry.train(full)
            child=Path(c.log.log_dir)/'child';full=Path(c.log.log_dir)/'full'
            self.assertEqual((child/'model_initial.eqx').read_bytes(),(parent/'model_final.eqx').read_bytes())
            self.assertEqual((child/'optimizer_initial.eqx').read_bytes(),(parent/'optimizer_final.eqx').read_bytes())
            template=model(c,jax.random.PRNGKey(1));opt=make_optimizer(c)
            for filename,tree in (('model_final.eqx',template),('optimizer_final.eqx',opt.init(eqx.filter(template,eqx.is_array)))):
                a=eqx.tree_deserialise_leaves(child/filename,tree);b=eqx.tree_deserialise_leaves(full/filename,tree)
                for u,v in zip(jax.tree.leaves(a),jax.tree.leaves(b),strict=True):
                    if eqx.is_array(u):np.testing.assert_allclose(u,v,rtol=1e-6,atol=1e-8)
            with np.load(child/'sampling.npz',allow_pickle=False) as a:
                with np.load(full/'sampling.npz',allow_pickle=False) as b:
                    for name in a.files:np.testing.assert_array_equal(a[name],b[name][3:])
            summary=json.loads((child/'summary.json').read_text())
            self.assertEqual(summary['start_step'],3);self.assertEqual(summary['new_updates'],3)
            self.assertGreater(summary['cumulative_wall_seconds'],summary['wall_seconds'])
            rows=[json.loads(s) for s in (child/'metrics.jsonl').read_text().splitlines()]
            self.assertEqual([r['step'] for r in rows],[3,4,5,6])

    def test_full_nonzero_soap_state_and_next_update_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory);c,model,optimizer,state,loss=parent_fixture(path)
            c.train.iterations=8;c.log.run_name='child'
            c.train.continuation=ConfigDict(dict(parent_run=str(path),kind='strict'))
            spec=continuation_spec(c)
            template=eqx.nn.Linear(3,2,key=jax.random.PRNGKey(91))
            imported,restored,start=restore_state(template,optimizer,spec)
            self.assertEqual(start,3)
            for a,b in zip(jax.tree.leaves((model,state)),jax.tree.leaves((imported,restored)),strict=True):
                if eqx.is_array(a):np.testing.assert_array_equal(a,b)
            self.assertTrue(any(np.issubdtype(np.asarray(a).dtype,np.integer) and np.any(np.asarray(a)>0)
                for a in jax.tree.leaves(state) if eqx.is_array(a)))
            left,ls=optimizer.update(eqx.filter_grad(loss)(model),state,model)
            right,rs=optimizer.update(eqx.filter_grad(loss)(imported),restored,imported)
            for a,b in zip(jax.tree.leaves((left,ls)),jax.tree.leaves((right,rs)),strict=True):
                if eqx.is_array(a):np.testing.assert_allclose(a,b,atol=1e-8,rtol=1e-6)

    def test_refine_is_explicit_and_cannot_change_optimizer_or_physics(self):
        with tempfile.TemporaryDirectory() as directory:
            c,_,_,_,_=parent_fixture(Path(directory));c.train.iterations=8
            c.train.continuation=ConfigDict(dict(parent_run=directory,kind='strict'))
            c.sample.Nx*=2;c.sample.Ny*=2
            with self.assertRaisesRegex(ValueError,'configuration'):continuation_spec(c)
            c.train.continuation.kind='refine';spec=continuation_spec(c)
            self.assertEqual(spec['resolution_before'],[96,40]);self.assertEqual(spec['resolution_after'],[192,80])
            c.train.soap_b2=.8
            with self.assertRaisesRegex(ValueError,'configuration'):continuation_spec(c)
            c.train.soap_b2=.9;c.physical.mobility*=2
            with self.assertRaisesRegex(ValueError,'configuration'):continuation_spec(c)

    def test_missing_complete_state_fails_without_reset(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory);c,_,_,_,_=parent_fixture(path)
            c.train.iterations=8;c.train.continuation=ConfigDict(dict(parent_run=directory,kind='strict'))
            (path/'optimizer_final.eqx').unlink()
            with self.assertRaises(FileNotFoundError):continuation_spec(c)


if __name__=='__main__':unittest.main()
