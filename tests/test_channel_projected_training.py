"""Reduced chemical system: physical gradients, finite-stage provenance and replay."""
import contextlib
from concurrent.futures import ThreadPoolExecutor
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from ml_collections import ConfigDict

from examples.channel_nsch2d.causal import make_causal_fn,audit_comparison
from examples.channel_nsch2d.data import geometry_data
from examples.channel_nsch2d.initialization import model_initialization_spec,audit_model_initialization
from examples.channel_nsch2d.losses import make_loss_fn,part_names
from examples.channel_nsch2d.network import build_network,encode,fields,potentials
from examples.channel_nsch2d.problem import validate_config,identity
from examples.channel_nsch2d.temporal import sample_continuous
from tests.test_channel_projected_fields import config
from tests.test_channel_time_jvp import config as parent_config


class ChannelProjectedTrainingTests(unittest.TestCase):
    def test_full_physical_gradient_eight_parts_and_global40_accumulation(self):
        with jax.enable_x64(True):
            c=config();validate_config(c)
            raw=[geometry_data(a,c)[0]for a in ((.12,.08),(.06,.13))]
            batch={k:jnp.asarray(np.stack([d[k]for d in raw]))for k in raw[0]}
            net=build_network(c,jax.random.PRNGKey(31))
            times=jax.jit(lambda k:sample_continuous(k,c))(jax.random.PRNGKey(7)).astype(jnp.float64)
            self.assertEqual(part_names(c),('phase','momentum_x','momentum_y','wall_phase','wall_mu','outlet_traction','outlet_phase','outlet_mu'))
            loss=make_loss_fn(c)
            (value,parts),gradient=eqx.filter_jit(eqx.filter_value_and_grad(loss,has_aux=True))(net,batch,times)
            self.assertEqual(parts.shape,(8,))
            actual,vector,grad,_=eqx.filter_jit(make_causal_fn(c))(net,batch,times,jnp.asarray(0.))
            np.testing.assert_allclose(actual,value,rtol=2e-12,atol=2e-12)
            np.testing.assert_allclose(vector,parts,rtol=2e-12,atol=2e-12)
            for a,b in zip(jax.tree.leaves(grad),jax.tree.leaves(gradient),strict=True):
                np.testing.assert_allclose(a,b,rtol=2e-10,atol=2e-11)
            evaluate=eqx.filter_jit(lambda n:loss(n,batch,times)[0]);h=1e-5
            for index in ((0,0),(1,0)):
                plus=eqx.tree_at(lambda n:n.output.weight,net,net.output.weight.at[index].add(h))
                minus=eqx.tree_at(lambda n:n.output.weight,net,net.output.weight.at[index].add(-h))
                fd=float((evaluate(plus)-evaluate(minus))/(2*h));self.assertGreater(abs(fd),1e-8)
                np.testing.assert_allclose(gradient.output.weight[index],fd,rtol=3e-5,atol=3e-6)

    def test_schema_guards_and_config_validation_without_JAX_import(self):
        c=config();validate_config(c)
        for field in ('kind','boundary','query'):
            bad=config();bad.chemical_constraint[field]='unknown'
            with self.assertRaisesRegex(ValueError,'Chemical constraint'):validate_config(bad)
        c=config();del c.time_derivative
        with self.assertRaisesRegex(ValueError,'chemical projection'):validate_config(c)
        c=config()
        for fn in (fields,potentials):
            with self.assertRaisesRegex(ValueError,'Projected mu'):fn(None,None,None,None,None,c)
        result=subprocess.run([sys.executable,'-c',
            "import sys;import examples.channel_nsch2d.initialization;assert 'jax' not in sys.modules"],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_actual_parent_and_model_only_stage_training_replay_and_tamper(self):
        from examples.channel_nsch2d import train as entry
        from examples.channel_nsch2d.cohort import prepare
        from examples.channel_nsch2d.validate_run import validate
        from examples.channel_nsch2d.diagnose_phase import diagnose
        from examples.channel_nsch2d.diagnose_time import diagnose as diagnose_time
        c=parent_config();c.train.iterations=3;c.operator.online.refresh_every=1
        # This width4 plumbing fixture is not the production width128 model.
        # Keep its coordinate map smooth enough for the unchanged64-point
        # independent flux quadrature; parent and child use the same bandwidth.
        c.arch.spatial_scale=.25
        c.operator.online.refresh_size=2;c.log.eval_every=3;c.log.checkpoint_every=3
        with tempfile.TemporaryDirectory()as directory,jax.enable_x64(True):
            root=Path(directory);c.reference.root=str(root);c.log.log_dir=str(root/'runs');c.log.run_name='parent'
            c.operator.validation_manifest=str(root/'reserved/manifest.json')
            prepare(c,c.operator.validation_manifest,4,(96,40))
            reference=root/'reference.npz';target=np.ones((13,3,5));target[...,0]=.5
            np.savez(reference,points=np.array([[.2,.3],[.8,.5],[2.,.5]]),time=np.linspace(0,6,13),amplitudes=np.array([.12,.08]),fields=target)
            (root/'reference-validation.json').write_text(json.dumps(dict(status='passed',reference=str(reference),identity=identity(c))))
            def executor(**kwargs):return ThreadPoolExecutor(max_workers=kwargs['max_workers'])
            with patch('examples.channel_nsch2d.online.ProcessPoolExecutor',executor),contextlib.redirect_stdout(io.StringIO()):
                entry.train(c);parent=Path(c.log.log_dir)/c.log.run_name;validate(parent)
                reduced=ConfigDict(c.to_dict());reduced.chemical_constraint=config().chemical_constraint
                reduced.train.initialization=dict(kind='physics_model',source_run=str(parent),optimizer='fresh')
                reduced.train.lr_init=1e-4;reduced.log.run_name='projected'
                validate_config(reduced);entry.train(reduced)
                run=Path(reduced.log.log_dir)/reduced.log.run_name;validate(run);diagnose(run,None)
                diagnostic=diagnose_time(run,root/'time-audit',8,881)
            self.assertEqual(audit_model_initialization(run,reduced)['source_physics_updates'],3)
            paired=audit_comparison(run,parent)
            self.assertEqual(paired['status'],'physics_model_initialization_fresh_optimizer_and_actual_input_streams_passed')
            self.assertTrue(paired['all_actual_sampling_arrays_exact'])
            self.assertEqual(json.loads((run/'validation.json').read_text())['model_initialization']['new_updates'],3)
            self.assertFalse(json.loads((run/'phase-balance.json').read_text())['spatial_mu_AD_claimed'])
            for row in diagnostic['cases']:
                with np.load(row['source'],allow_pickle=False)as f:self.assertEqual(f['parts'].shape,(8,8))
            for field in ('arch','physical'):
                bad=ConfigDict(reduced.to_dict())
                if field=='arch':bad.arch.width*=2
                else:bad.physical.mobility*=2
                with self.assertRaises(ValueError):model_initialization_spec(bad)
            original=(run/'optimizer_initial.eqx').read_bytes()
            (run/'optimizer_initial.eqx').write_bytes((parent/'optimizer_final.eqx').read_bytes())
            with self.assertRaisesRegex(ValueError,'fresh factory'):audit_model_initialization(run,reduced)
            (run/'optimizer_initial.eqx').write_bytes(original)


if __name__=='__main__':unittest.main()
