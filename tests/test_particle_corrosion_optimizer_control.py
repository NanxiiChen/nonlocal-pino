"""Optimizer controls share cold models/traces, never incompatible states."""
import json
from pathlib import Path
import tempfile
import unittest

import equinox as eqx
import jax
import numpy as np
from ml_collections import ConfigDict

from examples.particle_corrosion2d.configs.train_physical import get_config as physical_config
from examples.particle_corrosion2d.configs.train_physical_causal_early_soap import get_config as soap_config
from examples.particle_corrosion2d.sampling import initial_layer_times
from examples.particle_corrosion2d.strategies import load_strategy_control,verify_cold_initialization
from nonlocal_pinn.training.optimizers import make_optimizer


class OptimizerControlTests(unittest.TestCase):
    def test_optimizer_swap_replays_model_and_draws_but_owns_its_state(self):
        prior=physical_config();prior.train.iterations=6
        current=soap_config();current.train.iterations=3
        net=eqx.nn.Linear(2,2,key=jax.random.key(17))
        adam_state=make_optimizer(prior).init(eqx.filter(net,eqx.is_array))
        soap_state=make_optimizer(current).init(eqx.filter(net,eqx.is_array))
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);run=root/'run';run.mkdir()
            inputs=root/'inputs';inputs.mkdir();reference=root/'reference';reference.mkdir()
            (run/'config.json').write_text(prior.to_json())
            (run/'runtime.json').write_text(json.dumps(dict(seed=17,inputs=str(inputs),reference=str(reference))))
            (run/'summary.json').write_text(json.dumps(dict(steps=6)))
            eqx.tree_serialise_leaves(run/'model_initial.eqx',net)
            eqx.tree_serialise_leaves(run/'model_final.eqx',net)
            eqx.tree_serialise_leaves(run/'optimizer_initial.eqx',adam_state)
            rng=np.random.default_rng(17);shape=(6,prior.operator.condition_batch)
            trace=dict(condition_ids=rng.integers(64,size=shape,dtype=np.int32),
                       pool_ids=rng.integers(prior.sample.K_pool,size=shape,dtype=np.int32),
                       times=rng.uniform(0,1,size=(6,32)).astype(np.float32))
            np.savez(run/'sampling.npz',**trace)
            _,actual=load_strategy_control(current,run,inputs,reference)
            np.testing.assert_array_equal(actual['condition_ids'],trace['condition_ids'][:3])
            np.testing.assert_array_equal(actual['pool_ids'],trace['pool_ids'][:3])
            np.testing.assert_array_equal(actual['times'],initial_layer_times(trace['times'][:3],current))
            self.assertFalse(verify_cold_initialization(net,soap_state,current,run,True))
            self.assertTrue(verify_cold_initialization(net,adam_state,prior,run,True))
            with self.assertRaisesRegex(ValueError,'same optimizer'):
                verify_cold_initialization(net,soap_state,current,run,False)
            changed=eqx.tree_at(lambda model:model.bias,net,net.bias+.1)
            with self.assertRaisesRegex(ValueError,'saved initialization'):
                verify_cold_initialization(changed,soap_state,current,run,True)
            current.arch.time_coordinate=ConfigDict(dict(kind='logarithmic',time_scale=.00284))
            load_strategy_control(current,run,inputs,reference)
            current.arch.width+=4
            with self.assertRaisesRegex(ValueError,'non-strategy'):
                load_strategy_control(current,run,inputs,reference)
            current.arch.width-=4
            current.train.lr_init=2e-3
            with self.assertRaisesRegex(ValueError,'non-strategy'):
                load_strategy_control(current,run,inputs,reference)
            current.train.lr_init=prior.train.lr_init
            prior.sample.temporal=ConfigDict(current.sample.temporal.to_dict())
            (run/'config.json').write_text(prior.to_json())
            trace['times']=initial_layer_times(trace['times'],prior)
            np.savez(run/'sampling.npz',**trace)
            _,actual=load_strategy_control(current,run,inputs,reference)
            np.testing.assert_array_equal(actual['times'],trace['times'][:3])
            current.sample.temporal.early_end*=2
            with self.assertRaisesRegex(ValueError,'same initial-layer'):
                load_strategy_control(current,run,inputs,reference)
            current.sample.temporal.early_end/=2
            current.loss.gradnorm_norm='l2';current.train.gradnorm_update_every=1
            load_strategy_control(current,run,inputs,reference)
            current.loss.data_warmup_steps=0
            current.loss.use_causal=False
            current.sample.horizon_curriculum=ConfigDict(dict(kind='exponential',initial_fraction=.02,ramp_steps=2))
            _,actual=load_strategy_control(current,run,inputs,reference)
            expected=trace['times'][:3].copy();expected[0,8:]*=np.float32(.02)
            np.testing.assert_array_equal(actual['times'],expected)
            del current.sample.horizon_curriculum
            current.train.gradnorm_update_every=0
            from examples.particle_corrosion2d.strategies import validate_strategies
            with self.assertRaisesRegex(ValueError,'positive integer refresh'):
                validate_strategies(current)


if __name__=='__main__':unittest.main()
