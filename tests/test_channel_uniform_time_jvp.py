"""Uniform timeJVP controls retain cold state and actual geometry/RNG streams."""
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
from ml_collections import ConfigDict
import numpy as np

from examples.channel_nsch2d.configs.train_operator_width128_gmls_time40_jvp_uniform_online import get_config
from examples.channel_nsch2d.causal import comparison_config,audit_comparison
from examples.channel_nsch2d.data import geometry_data
from examples.channel_nsch2d.losses import make_loss_fn
from examples.channel_nsch2d.network import build_network
from examples.channel_nsch2d.problem import identity,validate_config
from examples.channel_nsch2d.temporal import sample_continuous
from examples.channel_nsch2d.time_batching import make_accumulated_fn
from tests.test_channel_time_jvp import config as adaptive_config


def config():
    c=adaptive_config();del c.causal
    return c


class ChannelUniformTimeJVPTests(unittest.TestCase):
    def test_strict_uniform_matching_and_preserved_adaptive_rules(self):
        c=get_config();validate_config(c);self.assertNotIn('causal',c)
        self.assertNotIn('chemical_constraint',c);self.assertNotIn('initialization',c.train)
        with tempfile.TemporaryDirectory()as directory:
            root=Path(directory);saved=ConfigDict(c.to_dict());del saved.time_derivative
            saved.time_discretization=dict(kind='bdf2',dt=.025)
            for name in ('sampling.npz','prediction.npz','known_inputs.npz','known_inputs_final.npz','model_initial.eqx','optimizer_initial.eqx'):
                (root/name).touch()
            (root/'config.json').write_text(saved.to_json())
            (root/'summary.json').write_text(json.dumps(dict(status='completed',iterations=c.train.iterations)))
            (root/'validation.json').write_text(json.dumps(dict(status='replay_and_hard_constraints_passed')))
            self.assertEqual(comparison_config(c,root),json.loads(saved.to_json()))
            bad=get_config();bad.train.lr_init*=2
            with self.assertRaisesRegex(ValueError,'only one explicit'):comparison_config(bad,root)
            saved.causal=adaptive_config().causal;(root/'config.json').write_text(saved.to_json())
            with self.assertRaisesRegex(ValueError,'uniformBDF'):comparison_config(c,root)

    def test_complete_uniform40_gradient_matches_all_time_loss(self):
        with jax.enable_x64(True):
            c=config();validate_config(c)
            raw=[geometry_data(a,c)[0]for a in ((.12,.08),(.06,.13))]
            batch={k:jnp.asarray(np.stack([d[k]for d in raw]))for k in raw[0]}
            net=build_network(c,jax.random.PRNGKey(41));times=jax.jit(lambda k:sample_continuous(k,c))(jax.random.PRNGKey(7)).astype(jnp.float64)
            loss=make_loss_fn(c)
            (total,parts),gradient=eqx.filter_jit(eqx.filter_value_and_grad(loss,has_aux=True))(net,batch,times)
            (actual,vector),grad=eqx.filter_jit(make_accumulated_fn(c,loss))(net,batch,times)
            np.testing.assert_allclose(actual,total,rtol=2e-12,atol=2e-12)
            np.testing.assert_allclose(vector,parts,rtol=2e-12,atol=2e-12)
            for a,b in zip(jax.tree.leaves(grad),jax.tree.leaves(gradient),strict=True):
                np.testing.assert_allclose(a,b,rtol=2e-10,atol=2e-11)

    def test_actual_matched_cold_bdf_and_uniform_jvp_training_replay(self):
        from examples.channel_nsch2d import train as entry
        from examples.channel_nsch2d.cohort import prepare
        from examples.channel_nsch2d.validate_run import validate
        from examples.channel_nsch2d.diagnose_time import diagnose
        c=config();del c.time_derivative;c.time_discretization=dict(kind='bdf2',dt=.025)
        c.train.iterations=3;c.operator.online.refresh_every=1;c.operator.online.refresh_size=2
        c.log.eval_every=3;c.log.checkpoint_every=3;c.arch.spatial_scale=.25
        with tempfile.TemporaryDirectory()as directory,jax.enable_x64(True):
            root=Path(directory);c.reference.root=str(root);c.log.log_dir=str(root/'runs');c.log.run_name='uniform-bdf'
            c.operator.validation_manifest=str(root/'reserved/manifest.json');prepare(c,c.operator.validation_manifest,4,(96,40))
            reference=root/'reference.npz';target=np.ones((13,3,5));target[...,0]=.5
            np.savez(reference,points=np.array([[.2,.3],[.8,.5],[2.,.5]]),time=np.linspace(0,6,13),amplitudes=np.array([.12,.08]),fields=target)
            (root/'reference-validation.json').write_text(json.dumps(dict(status='passed',reference=str(reference),identity=identity(c))))
            def executor(**kwargs):return ThreadPoolExecutor(max_workers=kwargs['max_workers'])
            with patch('examples.channel_nsch2d.online.ProcessPoolExecutor',executor),contextlib.redirect_stdout(io.StringIO()):
                entry.train(c);control=Path(c.log.log_dir)/c.log.run_name;validate(control)
                jvp=ConfigDict(c.to_dict());del jvp.time_discretization
                jvp.time_derivative=dict(kind='jvp',sampling='continuous_stratified');jvp.log.run_name='uniform-jvp'
                comparison_config(jvp,control);entry.train(jvp)
                run=Path(jvp.log.log_dir)/jvp.log.run_name;validate(run)
                report=audit_comparison(run,control)
                diagnostic=diagnose(run,root/'continuous-audit',8,997)
            self.assertEqual(report['status'],'matched_cold_state_geometry_stream_and_independent_continuous_time_replay_passed')
            self.assertTrue(report['initial_model_and_complete_optimizer_exact'])
            self.assertTrue(report['initial_and_final_input_banks_exact'])
            self.assertFalse((run/'causal.jsonl').exists());self.assertFalse((run/'causal-controller.jsonl').exists())
            self.assertNotEqual((run/'model_initial.eqx').read_bytes(),(run/'model_final.eqx').read_bytes())
            self.assertEqual(diagnostic['part_names'],['phase','chemical','momentum_x','momentum_y','wall_phase','wall_mu','outlet_traction','outlet_phase','outlet_mu'])


if __name__=='__main__':unittest.main()
