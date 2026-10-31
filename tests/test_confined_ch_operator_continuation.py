"""Real SOAP/PDE continuation through serialization and a refresh boundary."""
import json
from pathlib import Path
import tempfile
import unittest
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from ml_collections import ConfigDict
from examples.confined_ch2d.configs.train_operator_smallscale10_512_fourier import get_config
from examples.confined_ch2d.operator_continuation import continuation_spec,initial_state
from examples.confined_ch2d.operator_training import build_network,make_loss,operator_temporal_samples,fields
from examples.confined_ch2d.operator_problem import fixed_initial
from nonlocal_pinn.training.optimizers import make_optimizer
from test_confined_ch_operator import small_fixture


class OperatorContinuationTests(unittest.TestCase):
    def test_intermediate_marker_requires_both_complete_payloads(self):
        from unittest.mock import patch
        from examples.confined_ch2d.operator_continuation import save_checkpoint
        with tempfile.TemporaryDirectory() as directory:
            run=Path(directory);(run/'config.json').write_text('{}')
            model={'weight':np.ones((2,2))};state={'count':np.asarray(6,dtype=np.int32)}
            serialise=eqx.tree_serialise_leaves;calls=[]
            def interrupted(stream,value):
                calls.append(1)
                if len(calls)==2:raise OSError('Injected interrupted optimizer write')
                serialise(stream,value)
            with patch('examples.confined_ch2d.operator_continuation.eqx.tree_serialise_leaves',interrupted):
                with self.assertRaisesRegex(OSError,'Injected'):save_checkpoint(run,model,state,6)
            self.assertTrue((run/'ckpt_000006.eqx').exists())
            self.assertFalse((run/'checkpoint_000006.json').exists())
            self.assertFalse((run/'optimizer_000006.eqx').exists())
            self.assertTrue((run/'optimizer_000006.eqx.partial').exists())
            marker=save_checkpoint(run,model,state,7)
            r=json.loads(marker.read_text());self.assertEqual(r['step'],7)
            np.testing.assert_array_equal(eqx.tree_deserialise_leaves(run/r['model'],model)['weight'],model['weight'])
            np.testing.assert_array_equal(eqx.tree_deserialise_leaves(run/r['optimizer'],state)['count'],state['count'])
            with self.assertRaises(FileExistsError):save_checkpoint(run,model,state,7)

    def test_declared_learning_rate_reduction_keeps_adaptive_state_and_counter(self):
        with jax.enable_x64(True),tempfile.TemporaryDirectory() as directory:
            c=get_config();c.train.seed=42;c.train.iterations=3;c.sample.Nt=2
            c.arch.width=4;c.arch.heads=1;c.arch.condition_depth=1;c.arch.decoder_depth=1
            c.arch.spatial_features=2;c.arch.time_features=2;c.operator.condition_batch=1
            _,data=small_fixture();data=dict(data)
            data['initial']=jnp.where(data['volumes']>0,fixed_initial(data['points'],c,xp=jnp),0.)
            batch=jax.tree.map(lambda a:a[None],data);times=jnp.array([0.,.4,2.1])
            net=build_network(c,jax.random.key(42));opt=make_optimizer(c)
            state=opt.init(eqx.filter(net,eqx.is_array));loss=make_loss(c)
            @eqx.filter_jit
            def advance(model,state):
                _,g=eqx.filter_value_and_grad(loss)(model,batch,times)
                delta,state=opt.update(g,state,model)
                return eqx.apply_updates(model,delta),state
            for _ in range(3):net,state=advance(net,state)
            parent=Path(directory)
            eqx.tree_serialise_leaves(parent/'model_final.eqx',net)
            eqx.tree_serialise_leaves(parent/'optimizer_final.eqx',state)
            (parent/'config.json').write_text(c.to_json())
            (parent/'summary.json').write_text(json.dumps(dict(final_metrics=dict(step=3),
                checkpoint='model_final.eqx',wall_seconds=1.)))
            child=ConfigDict(c.to_dict());child.train.iterations=8
            child.train.lr_init*=.3;child.train.lr_min*=.3
            child.train.continuation=ConfigDict(dict(parent_run=str(parent),learning_rate_factor=.3))
            spec=continuation_spec(child);fine_opt=make_optimizer(child)
            restored,restored_state,start,_=initial_state(net,fine_opt,spec)
            self.assertEqual(start,3);self.assertEqual(spec['learning_rate_scaling']['factor'],.3)
            self.assertTrue(any(eqx.is_array(a) and a.shape==() and np.issubdtype(a.dtype,np.integer)
                and int(a)==3 for a in jax.tree.leaves(restored_state)))
            gradient=eqx.filter_jit(eqx.filter_grad(loss))(restored,batch,times)
            original_delta,next_state=opt.update(gradient,state,restored)
            fine_delta,next_fine_state=fine_opt.update(gradient,restored_state,restored)
            # Optax's int32 counter makes this scalar schedule float32 even for
            # float64 parameters. Check its actual scalar ratio, not exact .3.
            def rate(config):
                schedule=optax.exponential_decay(config.train.lr_init,config.train.lr_decay_every,
                    config.train.lr_decay_rate,end_value=config.train.lr_min)
                return float(schedule(jnp.asarray(start,dtype=jnp.int32)))
            ratio=rate(child)/rate(c)
            self.assertAlmostEqual(ratio,.3,places=6)
            magnitude=0.
            for a,b in zip(jax.tree.leaves(original_delta),jax.tree.leaves(fine_delta),strict=True):
                if eqx.is_array(a):
                    np.testing.assert_allclose(b,ratio*a,rtol=1e-11,atol=1e-14)
                    magnitude+=float(jnp.linalg.norm(a))
            self.assertGreater(magnitude,1e-8)
            for a,b in zip(jax.tree.leaves(next_state),jax.tree.leaves(next_fine_state),strict=True):
                if eqx.is_array(a):np.testing.assert_array_equal(a,b)
            for factor in (True,0.,1.,float('nan')):
                broken=ConfigDict(child.to_dict());broken.train.continuation.learning_rate_factor=factor
                with self.assertRaisesRegex(ValueError,'learning_rate_factor'):continuation_spec(broken)
            broken=ConfigDict(child.to_dict());broken.train.lr_min=c.train.lr_min
            with self.assertRaisesRegex(ValueError,'scale train.lr_min'):continuation_spec(broken)
            broken=ConfigDict(child.to_dict());broken.train.soap_b1=.8
            with self.assertRaisesRegex(ValueError,'optimizer/runtime'):continuation_spec(broken)
            broken=ConfigDict(child.to_dict());broken.train.continuation.unrecognized=True
            with self.assertRaisesRegex(ValueError,'Unknown continuation'):continuation_spec(broken)

    def test_real_state_and_global_time_draws_match_uninterrupted_steps(self):
        with jax.enable_x64(True),tempfile.TemporaryDirectory() as directory:
            c=get_config();c.train.seed=42;c.train.iterations=3;c.sample.Nt=2
            c.arch.width=4;c.arch.heads=1;c.arch.condition_depth=1;c.arch.decoder_depth=1
            c.arch.spatial_features=2;c.arch.time_features=2;c.operator.condition_batch=1
            _,data=small_fixture();data=dict(data)
            data['initial']=jnp.where(data['volumes']>0,fixed_initial(data['points'],c,xp=jnp),0.)
            batch=jax.tree.map(lambda a:a[None],data)
            keys=jax.random.split(jax.random.key(42),3);original=build_network(c,keys[0]);opt=make_optimizer(c)
            self.assertIsNone(continuation_spec(c))
            net,state,start,info=initial_state(original,opt,None)
            self.assertEqual(start,0);self.assertIsNone(info)
            def make_step(optimizer,config):
                loss=make_loss(config)
                @eqx.filter_jit
                def step(model,state,times):
                    _,g=eqx.filter_value_and_grad(loss)(model,batch,times)
                    delta,state=optimizer.update(g,state,model)
                    return eqx.apply_updates(model,delta),state
                return step
            step=make_step(opt,c);parent=Path(directory)
            for i in range(1,9):
                times,_=operator_temporal_samples(c,i,jax.random.fold_in(keys[2],i))
                net,state=step(net,state,times)
                if i==3:
                    eqx.tree_serialise_leaves(parent/'model_final.eqx',net)
                    eqx.tree_serialise_leaves(parent/'optimizer_final.eqx',state)
                    parent_fields=fields(net,data,jnp.array([0.,.4,3.]),c)
            (parent/'config.json').write_text(c.to_json())
            (parent/'summary.json').write_text(json.dumps(dict(final_metrics=dict(step=3),
                checkpoint='model_final.eqx',wall_seconds=1.25)))
            child=ConfigDict(c.to_dict());child.train.iterations=8
            child.train.continuation=ConfigDict(dict(parent_run=str(parent)))
            spec=continuation_spec(child);opt2=make_optimizer(child)
            resumed,state2,start,info=initial_state(original,opt2,spec)
            self.assertEqual(start,3);self.assertEqual(info['new_updates'],5)
            np.testing.assert_array_equal(fields(resumed,data,jnp.array([0.,.4,3.]),child),parent_fields)
            step2=make_step(opt2,child)
            for i in range(start+1,child.train.iterations+1):
                times,_=operator_temporal_samples(child,i,jax.random.fold_in(keys[2],i))
                resumed,state2=step2(resumed,state2,times)
            for first,second in ((net,resumed),(state,state2)):
                for a,b in zip(jax.tree.leaves(first),jax.tree.leaves(second),strict=True):
                    if eqx.is_array(a):np.testing.assert_allclose(a,b,rtol=1e-11,atol=1e-12)
            broken=ConfigDict(child.to_dict());broken.arch.spatial_scale=3.
            with self.assertRaisesRegex(ValueError,'numerical configuration'):continuation_spec(broken)
            broken=ConfigDict(child.to_dict());broken.train.lr_init=.0001
            with self.assertRaisesRegex(ValueError,'optimizer/runtime'):continuation_spec(broken)
            broken=ConfigDict(child.to_dict());broken.train.iterations=3
            with self.assertRaisesRegex(ValueError,'larger total'):continuation_spec(broken)


    def test_trainer_writes_global_counter_parent_replay_and_cumulative_timing(self):
        from unittest.mock import patch
        from examples.confined_ch2d.operator_training import train_operator
        with jax.enable_x64(True),tempfile.TemporaryDirectory() as directory:
            c=get_config();c.train.seed=42;c.train.iterations=3;c.sample.Nt=2
            c.arch.width=4;c.arch.heads=1;c.arch.condition_depth=1;c.arch.decoder_depth=1
            c.arch.spatial_features=2;c.arch.time_features=2
            c.operator.train_conditions=2;c.operator.test_conditions=1;c.operator.condition_batch=1
            c.operator.precompute.reserved_test_draws=(2,);c.operator.precompute.monitor_train_indices=(0,1)
            c.log.log_dir=directory;c.log.run_name='parent';c.log.eval_every=3;c.log.checkpoint_every=3
            _,data=small_fixture();data=dict(data)
            data['initial']=jnp.where(data['volumes']>0,fixed_initial(data['points'],c,xp=jnp),0.)
            keys=jax.random.split(jax.random.key(42),3);template=build_network(c,keys[0])
            times=np.linspace(0,c.physical.terminal_time,c.reference.snapshots)
            target=np.asarray(fields(template,data,jnp.asarray(times),c))
            ref=dict(points=np.asarray(data['points']),volumes=np.asarray(data['volumes']),time=times,
                phi=target[...,0],mu=target[...,1],holes=np.array([[.3,.4,.05],[.7,.6,.06]]),
                mesh_points=np.asarray(data['points'][:3]),mesh_cells=np.array([[0,1,2]]))
            refs=[dict(ref) for _ in range(3)]
            bank=jax.tree.map(lambda a:jnp.stack([a,a,a]),data)
            manifest=dict(train_indices=[0,1],test_indices=[2])
            with patch('examples.confined_ch2d.operator_training.load_bank',return_value=(manifest,refs,bank)):
                parent=train_operator(c)
                child=ConfigDict(c.to_dict());child.train.iterations=8;child.log.run_name='child'
                child.train.continuation=ConfigDict(dict(parent_run=str(parent)))
                run=train_operator(child)
            history=[json.loads(line) for line in (run/'metrics.jsonl').read_text().splitlines()]
            self.assertEqual([r['step'] for r in history],[3,6,8])
            with np.load(run/'training_draws.npz') as f:draws=f['condition_ids']
            expected=np.array([jax.random.randint(jax.random.fold_in(keys[1],i),(1,),0,2) for i in range(4,9)])
            np.testing.assert_array_equal(draws,expected)
            summary=json.loads((run/'summary.json').read_text());previous=json.loads((parent/'summary.json').read_text())
            self.assertEqual(summary['start_step'],3);self.assertEqual(summary['new_updates'],5)
            self.assertEqual(summary['cumulative_wall_seconds'],previous['wall_seconds']+summary['wall_seconds'])
            # Recover the exact intermediate step6 model AND adaptive SOAP
            # state, then finish steps7/8 across the existing refresh schedule.
            marker=json.loads((run/'checkpoint_000006.json').read_text())
            self.assertEqual(marker['step'],6)
            self.assertEqual(marker['kind'],'intermediate_full_state')
            resumed=eqx.tree_deserialise_leaves(run/marker['model'],template)
            optimizer=make_optimizer(child)
            state_template=optimizer.init(eqx.filter(template,eqx.is_array))
            state=eqx.tree_deserialise_leaves(run/marker['optimizer'],state_template)
            self.assertTrue(any(eqx.is_array(x) and x.shape==() and np.issubdtype(x.dtype,np.integer)
                and int(x)==6 for x in jax.tree.leaves(state)))
            loss=make_loss(child)
            @eqx.filter_jit
            def advance(model,adaptive,times):
                _,gradient=eqx.filter_value_and_grad(loss)(model,jax.tree.map(lambda a:a[None],data),times)
                delta,adaptive=optimizer.update(gradient,adaptive,model)
                return eqx.apply_updates(model,delta),adaptive
            for i in (7,8):
                times,_=operator_temporal_samples(child,i,jax.random.fold_in(keys[2],i))
                resumed,state=advance(resumed,state,times)
            final_model=eqx.tree_deserialise_leaves(run/'model_final.eqx',template)
            final_state=eqx.tree_deserialise_leaves(run/'optimizer_final.eqx',state_template)
            for actual,expected in ((resumed,final_model),(state,final_state)):
                for x,y in zip(jax.tree.leaves(actual),jax.tree.leaves(expected),strict=True):
                    if eqx.is_array(x):np.testing.assert_allclose(x,y,rtol=1e-11,atol=1e-12)
            self.assertFalse(list(run.glob('*.partial')))
            a=eqx.tree_deserialise_leaves(parent/'model_final.eqx',template)
            b=eqx.tree_deserialise_leaves(run/'model_initial.eqx',template)
            for x,y in zip(jax.tree.leaves(a),jax.tree.leaves(b),strict=True):
                if eqx.is_array(x):np.testing.assert_array_equal(x,y)


if __name__=='__main__':unittest.main()
