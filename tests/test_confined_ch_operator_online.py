"""Deterministic asynchronous refresh, paired recovery and physical padding."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import tempfile
import time
import unittest
from unittest.mock import patch

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from ml_collections import ConfigDict

from examples.confined_ch2d.operator_continuation import continuation_spec,initial_state,save_checkpoint
from examples.confined_ch2d.operator_online import (
    sample_geometry,pad_input,pool_identities,refresh_count,RollingGeometryPool)
from examples.confined_ch2d.operator_problem import valid_holes
from examples.confined_ch2d.operator_training import build_network,fields,make_loss,train_operator
from examples.confined_ch2d.operator_precision import make_compute_optimizer
from test_confined_ch_operator import small_fixture


def fixture():
    c,data=small_fixture();c.train.seed=42;c.operator.train_conditions=4;c.train.iterations=15
    c.operator.metric='linear_stable_resolvent';c.operator.metric_workers=1
    c.operator.online=ConfigDict(dict(workers=2,refresh_size=2,refresh_every=2,
        prefetch_batches=2,max_points=7,max_neighbors=4,origin_step=3))
    c.train.continuation=ConfigDict(dict(parent_run='/tmp/unused',checkpoint='/tmp/unused/checkpoint_000003.json'))
    return c,data


class OnlineGeometryTests(unittest.TestCase):
    def test_canonical_online_loop_keeps_frozen_monitors_and_recovers_pool(self):
        self._canonical_online_recovery(origin=3)

    def test_cold_online_loop_initializes_at_zero_and_recovers_pool(self):
        self._canonical_online_recovery(origin=0)

    def _canonical_online_recovery(self,origin):
        import contextlib
        import io
        with jax.enable_x64(True),tempfile.TemporaryDirectory() as directory:
            c,raw=fixture();c.operator.train_conditions=2;c.operator.test_conditions=1;c.operator.condition_batch=1
            c.operator.local_validation_indices=()
            c.log.log_dir=directory;c.log.run_name='parent';c.log.eval_every=3;c.log.checkpoint_every=3
            c.sample.Nt=2;c.train.iterations=3
            data=pad_input({key:np.asarray(value) for key,value in raw.items()},c)
            settings=ConfigDict(c.operator.online.to_dict());settings.origin_step=origin
            del c.operator.online;del c.train.continuation
            keys=jax.random.split(jax.random.key(c.train.seed),3);template=build_network(c,keys[0])
            times=np.linspace(0,c.physical.terminal_time,c.reference.snapshots)
            target=np.asarray(fields(template,data,jnp.asarray(times),c))[:,:4]
            ref=dict(points=np.asarray(raw['points']),volumes=np.asarray(raw['volumes']),time=times,
                phi=target[...,0],mu=target[...,1],holes=np.array([[.3,.4,.05],[.7,.6,.06]]),
                mesh_points=np.asarray(raw['points'][:3]),mesh_cells=np.array([[0,1,2]]))
            refs=[dict(ref) for _ in range(3)];manifest=dict(train_indices=[0,1],test_indices=[2])
            def bank():return {key:jnp.stack([value,value,value]) for key,value in data.items()}
            def build(config,draw):
                changed=dict(data);changed['condition']=data['condition']+draw+2
                return changed,dict(draw_id=draw,holes=[[draw,0,1],[0,draw,1]])
            def executor(**kwargs):return ThreadPoolExecutor(max_workers=kwargs['max_workers'])
            with patch('examples.confined_ch2d.operator_training.load_bank',side_effect=lambda cc:(manifest,refs,bank())),\
                 patch('examples.confined_ch2d.operator_online.ProcessPoolExecutor',executor),\
                 patch('examples.confined_ch2d.operator_online.build_input',build),contextlib.redirect_stdout(io.StringIO()):
                parent=train_operator(c) if origin else None
                child=ConfigDict(c.to_dict());child.train.iterations=8;child.log.run_name='online'
                child.operator.online=settings
                if parent is not None:
                    child.train.continuation=ConfigDict(dict(parent_run=str(parent),checkpoint=str(parent/'checkpoint_000003.json')))
                run=train_operator(child)
                resumed=ConfigDict(child.to_dict());resumed.log.run_name='recovered'
                resumed.train.continuation=ConfigDict(dict(parent_run=str(run),checkpoint=str(run/'checkpoint_000006.json')))
                recovered=train_operator(resumed)
            if origin==0:
                receipt=json.loads((run/'initial_restore_audit.json').read_text())
                self.assertEqual(receipt['step'],0)
                self.assertTrue(all(value==0 for value in receipt['optimizer_integer_counters']))
                self.assertNotIn('continuation',json.loads((run/'config.json').read_text())['train'])
            marker=json.loads((run/'checkpoint_000006.json').read_text());self.assertIn('online_state',marker)
            for filename,tree in [('model_final.eqx',template),
                    ('optimizer_final.eqx',make_compute_optimizer(child).init(eqx.filter(template,eqx.is_array)))]:
                left=eqx.tree_deserialise_leaves(run/filename,tree);right=eqx.tree_deserialise_leaves(recovered/filename,tree)
                for a,b in zip(jax.tree.leaves(left),jax.tree.leaves(right),strict=True):
                    if eqx.is_array(a):np.testing.assert_allclose(a,b,rtol=2e-6,atol=1e-8)
            net=eqx.tree_deserialise_leaves(run/'model_final.eqx',template)
            expected=np.asarray(fields(net,data,jnp.asarray(times),child))[:,:4,0]
            with np.load(run/'prediction_000.npz',allow_pickle=False) as f:
                np.testing.assert_allclose(f['phi'],expected,rtol=1e-6,atol=1e-8)
            with np.load(run/'training_draws.npz',allow_pickle=False) as f:
                self.assertEqual(f['geometry_draw_ids'].shape,(8-origin,1));self.assertEqual(f['times'].shape,(8-origin,3))

    def test_stateless_valid_prior_and_padding_do_not_change_physical_loss(self):
        with jax.enable_x64(True):
            c,data=fixture()
            for draw in range(20):
                a=sample_geometry(c,draw);np.testing.assert_array_equal(a,sample_geometry(c,draw))
                self.assertTrue(valid_holes(a,c))
            self.assertFalse(np.array_equal(sample_geometry(c,0),sample_geometry(c,1)))
            raw={key:np.asarray(value) for key,value in data.items()}
            padded=pad_input(raw,c)
            net=build_network(c,jax.random.key(3));times=jnp.array([0.,.7,2.5])
            original=fields(net,data,times,c);extra=fields(net,padded,times,c)
            np.testing.assert_allclose(extra[:,:4],original,rtol=1e-6,atol=1e-7)
            original_loss=make_loss(c)(net,jax.tree.map(lambda a:a[None],data),times)
            padded_loss=make_loss(c)(net,jax.tree.map(lambda a:a[None],padded),times)
            np.testing.assert_allclose(padded_loss,original_loss,rtol=2e-6,atol=1e-8)
            self.assertTrue(np.all(padded['volumes'][4:]==0))
            self.assertTrue(np.all(padded['coefficients'][4:]==0))
            bad=dict(raw,phi=np.zeros(4))
            with self.assertRaisesRegex(ValueError,'eight known-input'):pad_input(bad,c)
            c.operator.online.max_points=3
            with self.assertRaisesRegex(ValueError,'overflow'):pad_input(raw,c)

    def test_worker_completion_order_and_restoration_preserve_pool_inputs(self):
        c,_=fixture();forbidden=np.full((1,2,3),-9.)
        def build(config,draw):
            time.sleep(.01 if draw%2==0 else 0.)
            return {'value':np.array([draw],dtype=np.float32)},dict(draw_id=draw,holes=[[draw,0,1],[0,draw,1]])
        def executor(**kwargs):return ThreadPoolExecutor(max_workers=kwargs['max_workers'])
        with tempfile.TemporaryDirectory() as directory,\
             patch('examples.confined_ch2d.operator_online.ProcessPoolExecutor',executor),\
             patch('examples.confined_ch2d.operator_online.build_input',build):
            run=Path(directory);bank={'value':jnp.full((6,1),-1.,dtype=jnp.float32)}
            frozen=jnp.take(bank['value'],jnp.array([0,4,5]),axis=0)
            pool=RollingGeometryPool(c,run,3,None,forbidden)
            try:
                bank=pool.start(bank)
                for step in range(4,8):bank=pool.refresh(bank,step)
                np.testing.assert_array_equal(np.asarray(bank['value'])[:4,0],[0,1,2,3])
                np.testing.assert_array_equal(np.asarray(bank['value'])[4:,0],[-1,-1])
                np.testing.assert_array_equal(frozen,np.full((3,1),-1.))
                state=pool.save_state(7)
                np.testing.assert_array_equal(pool.draw_ids,[0,1,2,3])
                self.assertEqual(refresh_count(7,c),2)
            finally:pool.close()
            restored=RollingGeometryPool(c,run,7,dict(online_state=str(state)),forbidden)
            try:
                bank=restored.start({'value':jnp.full((6,1),-1.,dtype=jnp.float32)})
                np.testing.assert_array_equal(np.asarray(bank['value'])[:4,0],[0,1,2,3])
                bank=restored.refresh(bank,8)
                np.testing.assert_array_equal(np.asarray(bank['value'])[:4,0],[4,5,2,3])
                np.testing.assert_array_equal(restored.draw_ids,pool_identities(3,4,2))
            finally:restored.close()
            rows=[json.loads(s) for s in (run/'online_refresh.jsonl').read_text().splitlines()]
            self.assertEqual([r['draw_ids'] for r in rows],[[0,1],[2,3],[4,5]])

    def test_paired_intermediate_restore_allows_only_declared_online_sampling(self):
        with jax.enable_x64(True),tempfile.TemporaryDirectory() as directory:
            c,_=fixture();del c.operator.online;del c.train.continuation;c.train.iterations=10
            parent=Path(directory);(parent/'config.json').write_text(c.to_json())
            model=build_network(c,jax.random.key(3));opt=make_compute_optimizer(c)
            state=opt.init(eqx.filter(model,eqx.is_array));marker=save_checkpoint(parent,model,state,3)
            row=dict(step=3,elapsed_seconds=1.2,conditions=[])
            (parent/'metrics.jsonl').write_text(json.dumps(row)+'\n')
            child=ConfigDict(c.to_dict());child.train.iterations=15
            child.train.continuation=ConfigDict(dict(parent_run=str(parent),checkpoint=str(marker)))
            child.operator.online=fixture()[0].operator.online
            spec=continuation_spec(child);restored,restored_state,start,_=initial_state(model,opt,spec)
            self.assertEqual(start,3);self.assertIn('sampling_change',spec)
            for a,b in zip(jax.tree.leaves((model,state)),jax.tree.leaves((restored,restored_state)),strict=True):
                if eqx.is_array(a):np.testing.assert_array_equal(a,b)
            child.physical.mobility*=2
            with self.assertRaisesRegex(ValueError,'numerical configuration'):continuation_spec(child)
            child.physical.mobility/=2;child.train.soap_b2=.7
            with self.assertRaisesRegex(ValueError,'optimizer/runtime'):continuation_spec(child)
            child.train.soap_b2=c.train.soap_b2;(parent/'optimizer_000003.eqx').unlink()
            with self.assertRaises(FileNotFoundError):continuation_spec(child)


if __name__=='__main__':unittest.main()
