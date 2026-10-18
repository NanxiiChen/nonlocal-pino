"""Completed-history selection, joint-gradient equivalence and exact restart."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import equinox as eqx
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np
from examples.fracture2d.history_network import HistoryReplay
from examples.fracture2d.learning_rate import make_rate_step, shared_value_and_grad, taskwise_value_and_grad
from examples.fracture2d.loading import run, validate
from examples.fracture2d.storage import atomic_json
from tests.test_fracture2d_budget_learning_rate import controlled, stop_after_drop, Toy, Replay


class EnergyReplayTests(unittest.TestCase):
    def test_completed_distinct_selection_masks_shortage_and_resamples_without_retracing(self):
        bank = HistoryReplay(capacity=16, nodes=4, fine_count=2)
        for i in range(13):
            bank = bank.append(dict(H=jnp.full(4,float(i+1)), fine_ids=jnp.array([i,i],jnp.int32)))
        traces = []
        @eqx.filter_jit
        def draw(count, key, enabled):
            traces.append(1)
            current = dict(H=jnp.full(4,100.), fine_ids=jnp.zeros(2,jnp.int32), replay_count=count)
            return bank.batch(current,key,enabled,old_count=5,fraction=.1)
        seen = set()
        for count in (0, 2, 12):
            for seed in range(4):
                batch = draw(jnp.asarray(count,jnp.int32),jax.random.PRNGKey(seed),jnp.asarray(True))
                weights = np.asarray(batch['weights'])
                old = np.asarray(batch['H'])[1:,0][weights[1:]>0]
                self.assertEqual(batch['H'].shape,(6,4))
                self.assertEqual(len(old),min(count,5))
                self.assertEqual(len(np.unique(old)),len(old))
                self.assertTrue(np.all((old>=1)&(old<=count)))
                self.assertTrue(np.isfinite(batch['H']).all())
                np.testing.assert_allclose(weights.sum(),1.,rtol=0,atol=1e-15)
                np.testing.assert_allclose(weights[0],.9 if count else 1.)
                if count:np.testing.assert_allclose(weights[1:][weights[1:]>0],.1/min(count,5))
                if count==12:seen.add(tuple(sorted(old)))
        self.assertGreater(len(seen),1)
        disabled=draw(jnp.asarray(12,jnp.int32),jax.random.PRNGKey(0),jnp.asarray(False))
        np.testing.assert_array_equal(disabled['weights'],[1.,0.,0.,0.,0.,0.])
        self.assertEqual(len(traces),1)

    def test_taskwise_gradient_and_one_clipped_adam_update_equal_joint_loss(self):
        import optax
        net=Toy(jnp.array([.1,-.7,2.]))
        batch=dict(H=jnp.arange(18,dtype=jnp.float64).reshape(6,3)/10,
                   fine_ids=jnp.zeros((6,2),jnp.int32),
                   weights=jnp.array([.9,.025,.025,.025,.025,0.]))
        def objective(model,data):
            return jnp.sum(data['weights']*jnp.mean((model.w[None]-data['H'])**2,axis=1))
        expected=eqx.filter_value_and_grad(objective)(net,batch)
        actual=eqx.filter_jit(taskwise_value_and_grad)(objective,net,batch)
        for a,b in zip(jax.tree.leaves(actual),jax.tree.leaves(expected),strict=True):
            np.testing.assert_allclose(a,b,rtol=1e-14,atol=1e-15)
        opt=optax.chain(optax.clip_by_global_norm(.01),optax.adam(1.))
        state=opt.init(net)
        joint=make_rate_step(objective,opt,old_count=5,fraction=.1)
        sequential=make_rate_step(objective,opt,old_count=5,fraction=.1,taskwise=True)
        args=(net,state,batch,Replay(),jax.random.PRNGKey(0),jnp.asarray(True),jnp.asarray(1e-4))
        for a,b in zip(jax.tree.leaves(joint(*args)),jax.tree.leaves(sequential(*args)),strict=True):
            np.testing.assert_allclose(a,b,rtol=1e-14,atol=1e-15)

    def test_real_energy_replay_rate_control_and_exact_endpoint_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);c=controlled(root)
            c.history.replay_enabled=True
            c.history.replay_sampling='completed_without_replacement'
            c.history.replay_batch=5
            c.history.replay_fraction=.1
            c.history.replay_every=1
            c.history.current_only_steps=0
            validate(c)
            with contextlib.redirect_stdout(io.StringIO()),patch(
                    'examples.fracture2d.loading.field_stop',side_effect=stop_after_drop):
                run(c,damage_solver='pinn')
            base=Path(c.output.root)
            rows=json.loads((base/'result.json').read_text())['curve']
            self.assertEqual([r['replay_snapshots'] for r in rows],[2,4])
            c.output.root=str(root/'interrupted')
            def interrupt(path,data):
                atomic_json(path,data)
                if path.name=='result.json' and 'curve' in data and len(data['curve'])==1:
                    raise InterruptedError('endpoint')
            with contextlib.redirect_stdout(io.StringIO()),patch(
                    'examples.fracture2d.loading.field_stop',side_effect=stop_after_drop),patch(
                    'examples.fracture2d.loading.atomic_json',side_effect=interrupt):
                with self.assertRaises(InterruptedError):run(c,damage_solver='pinn')
            c.output.resume=True
            with contextlib.redirect_stdout(io.StringIO()),patch(
                    'examples.fracture2d.loading.field_stop',side_effect=stop_after_drop):
                run(c,damage_solver='pinn')
            final=Path(c.output.root)/'state_0002'
            for name in ('network.eqx','optimizer.eqx','rate_control.json'):
                self.assertEqual((base/'state_0002'/name).read_bytes(),(final/name).read_bytes())
            with np.load(base/'state_0002/fields.npz') as a,np.load(final/'fields.npz') as b:
                for key in a.files:np.testing.assert_array_equal(a[key],b[key])
            c.history.replay_sampling='unknown'
            with self.assertRaisesRegex(ValueError,'explicit replay policy'):validate(c)

    def test_shared_coordinate_pullback_matches_full_network_joint_gradient(self):
        from examples.fracture2d.history_training import HistoryDamagePINN
        from examples.fracture2d.mesh import load_geometry
        from examples.fracture2d.fem_damage import make_energy_loss, make_shared_energy_loss
        with tempfile.TemporaryDirectory() as tmp:
            c=controlled(Path(tmp));m,op,_=load_geometry(c)
            p=HistoryDamagePINN(c,m,op)
            # Nonzero output weights exercise gradients all the way to queries.
            net=eqx.tree_at(lambda n:n.head.layers[-1].weight,p.net,
                jnp.full_like(p.net.head.layers[-1].weight,.02))
            histories=[jnp.array([1.,2.,3.,4.]),jnp.array([4.,2.,1.,3.])]
            contexts=[p.context(H) for H in histories]
            batch=dict(H=jnp.stack(histories),fine_ids=jnp.stack([x['fine_ids'] for x in contexts]),
                       weights=jnp.array([.9,.1]))
            arguments=dict(coarse=p.coarse,fine=p.fine,points=jnp.asarray(2*m.coordinates[:4]-1),
                operator=op,volumes=m.volumes,gc0=p.gc0,gc1=c.physical.Gc*c.physical.l0,floor=c.loss.floor)
            _,objective=make_energy_loss(**arguments)
            functions=make_shared_energy_loss(**arguments)
            expected=eqx.filter_jit(eqx.filter_value_and_grad(objective))(net,batch)
            actual=eqx.filter_jit(shared_value_and_grad)(functions,net,batch)
            for a,b in zip(jax.tree.leaves(actual),jax.tree.leaves(expected),strict=True):
                np.testing.assert_allclose(a,b,rtol=1e-11,atol=1e-13)
            self.assertGreater(float(jnp.linalg.norm(actual[1].query_mlp.layers[0].weight)),1e-10)
            self.assertGreater(float(jnp.linalg.norm(actual[1].query_norm.weight)),1e-10)


if __name__=='__main__':unittest.main()
