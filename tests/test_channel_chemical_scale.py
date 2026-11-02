"""Chemical row normalization changes only its loss, retaining physical mu scaling."""
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

from examples.channel_nsch2d.configs.train_operator_width128_gmls_time40_jvp_chemical_online import get_config
from examples.channel_nsch2d.configs.train_operator_width128_gmls_time40_jvp_uniform_online import get_config as baseline
from examples.channel_nsch2d.causal import comparison_config,audit_comparison
from examples.channel_nsch2d.data import geometry_data
from examples.channel_nsch2d.losses import make_loss_fn
from examples.channel_nsch2d.network import build_network,encode,fields
from examples.channel_nsch2d.problem import validate_config,identity
from examples.channel_nsch2d.temporal import sample_continuous
from examples.channel_nsch2d.time_batching import make_accumulated_fn
from tests.test_channel_uniform_time_jvp import config as small


class ChannelChemicalScaleTests(unittest.TestCase):
    def test_explicit_positive_scale_and_single_field_comparison(self):
        c=get_config();validate_config(c)
        self.assertEqual(c.loss.chemical_scale,4.);self.assertEqual(c.loss.chemical_residual_scale,1.)
        for value in (0.,-1.,float('nan'),float('inf'),True,[1.]):
            bad=baseline();bad.loss.chemical_residual_scale=value
            with self.assertRaisesRegex(ValueError,'chemical_residual_scale'):validate_config(bad)
        bad=get_config();bad.chemical_constraint=dict(kind='interior_gmls_projection',boundary='decoder_trace',query='mapped_bilinear')
        with self.assertRaisesRegex(ValueError,'unused'):validate_config(bad)
        with tempfile.TemporaryDirectory()as directory:
            root=Path(directory);saved=baseline()
            for name in ('sampling.npz','prediction.npz','known_inputs.npz','known_inputs_final.npz','model_initial.eqx','optimizer_initial.eqx'):(root/name).touch()
            (root/'config.json').write_text(saved.to_json())
            (root/'summary.json').write_text(json.dumps(dict(status='completed',iterations=c.train.iterations)))
            (root/'validation.json').write_text(json.dumps(dict(status='replay_and_hard_constraints_passed')))
            self.assertEqual(comparison_config(c,root),json.loads(saved.to_json()))
            bad=get_config();bad.loss.chemical_scale=1.
            with self.assertRaisesRegex(ValueError,'only chemical_residual_scale'):comparison_config(bad,root)
            bad=get_config();bad.train.lr_init*=2
            with self.assertRaisesRegex(ValueError,'only chemical_residual_scale'):comparison_config(bad,root)

    def test_primal_all_other_losses_gradient_decomposition_and40_accumulation(self):
        with jax.enable_x64(True):
            old=small();new=ConfigDict(old.to_dict());new.loss.chemical_residual_scale=1.
            raw=[geometry_data(a,old)[0]for a in ((.12,.08),(.06,.13))]
            batch={k:jnp.asarray(np.stack([d[k]for d in raw]))for k in raw[0]}
            d={k:a[0]for k,a in batch.items()};net=build_network(old,jax.random.PRNGKey(51))
            candidate=build_network(new,jax.random.PRNGKey(51))
            for a,b in zip(jax.tree.leaves(net),jax.tree.leaves(candidate),strict=True):
                if eqx.is_array(a):np.testing.assert_array_equal(a,b)
            ts=jax.jit(lambda k:sample_continuous(k,old))(jax.random.PRNGKey(11)).astype(jnp.float64)
            before=eqx.filter_jit(lambda n:fields(n,encode(n,d),d['points'],ts,d['amplitudes'],old))
            after=eqx.filter_jit(lambda n:fields(n,encode(n,d),d['points'],ts,d['amplitudes'],new))
            np.testing.assert_array_equal(before(net),after(candidate))
            prior=make_loss_fn(old);scaled=make_loss_fn(new)
            (value0,parts0),g0=eqx.filter_jit(eqx.filter_value_and_grad(prior,has_aux=True))(net,batch,ts)
            (value1,parts1),g1=eqx.filter_jit(eqx.filter_value_and_grad(scaled,has_aux=True))(net,batch,ts)
            ids=np.array([0,2,3,4,5,6,7,8]);np.testing.assert_array_equal(np.asarray(parts0)[ids],np.asarray(parts1)[ids])
            np.testing.assert_allclose(parts1[1],16*parts0[1],rtol=2e-12,atol=2e-12)
            np.testing.assert_allclose(value1,value0+15*parts0[1],rtol=2e-12,atol=2e-12)
            chemical=eqx.filter_jit(eqx.filter_grad(lambda n:prior(n,batch,ts)[1][1]))(net)
            for a,b,g in zip(jax.tree.leaves(g1),jax.tree.leaves(g0),jax.tree.leaves(chemical),strict=True):
                np.testing.assert_allclose(a,b+15*g,rtol=2e-9,atol=2e-10)
            (value,parts),gradient=eqx.filter_jit(make_accumulated_fn(new,scaled))(net,batch,ts)
            np.testing.assert_allclose(value,value1,rtol=2e-12,atol=2e-12)
            np.testing.assert_allclose(parts,parts1,rtol=2e-12,atol=2e-12)
            for a,b in zip(jax.tree.leaves(gradient),jax.tree.leaves(g1),strict=True):
                np.testing.assert_allclose(a,b,rtol=2e-10,atol=2e-11)
            objective=eqx.filter_jit(lambda n:scaled(n,batch,ts)[0]);h=1e-5
            for index in ((0,0),(1,0)):
                plus=eqx.tree_at(lambda n:n.output.weight,net,net.output.weight.at[index].add(h))
                minus=eqx.tree_at(lambda n:n.output.weight,net,net.output.weight.at[index].add(-h))
                fd=float((objective(plus)-objective(minus))/(2*h));self.assertGreater(abs(fd),1e-8)
                np.testing.assert_allclose(g1.output.weight[index],fd,rtol=3e-5,atol=3e-6)

    def test_actual_cold_training_replay_complete_inputs_and_scale_provenance(self):
        from examples.channel_nsch2d import train as entry
        from examples.channel_nsch2d.cohort import prepare
        from examples.channel_nsch2d.validate_run import validate
        from examples.channel_nsch2d.diagnose_time import diagnose
        c=small();c.train.iterations=3;c.operator.online.refresh_every=1;c.operator.online.refresh_size=2
        c.log.eval_every=3;c.log.checkpoint_every=3;c.arch.spatial_scale=.25
        with tempfile.TemporaryDirectory()as directory,jax.enable_x64(True):
            root=Path(directory);c.reference.root=str(root);c.log.log_dir=str(root/'runs');c.log.run_name='shared-scale'
            c.operator.validation_manifest=str(root/'reserved/manifest.json');prepare(c,c.operator.validation_manifest,4,(96,40))
            reference=root/'reference.npz';target=np.ones((13,3,5));target[...,0]=.5
            np.savez(reference,points=np.array([[.2,.3],[.8,.5],[2.,.5]]),time=np.linspace(0,6,13),amplitudes=np.array([.12,.08]),fields=target)
            (root/'reference-validation.json').write_text(json.dumps(dict(status='passed',reference=str(reference),identity=identity(c))))
            def executor(**kwargs):return ThreadPoolExecutor(max_workers=kwargs['max_workers'])
            with patch('examples.channel_nsch2d.online.ProcessPoolExecutor',executor),contextlib.redirect_stdout(io.StringIO()):
                entry.train(c);control=Path(c.log.log_dir)/c.log.run_name;validate(control)
                candidate=ConfigDict(c.to_dict());candidate.loss.chemical_residual_scale=1.;candidate.log.run_name='residual-scale'
                comparison_config(candidate,control);entry.train(candidate)
                run=Path(candidate.log.log_dir)/candidate.log.run_name;validate(run)
                audit=audit_comparison(run,control);diagnostic=diagnose(run,root/'continuous',8,999)
            self.assertEqual(audit['status'],'matched_cold_state_all_inputs_and_chemical_row_scaling_only_passed')
            self.assertTrue(audit['all_actual_sampling_arrays_exact']);self.assertEqual(audit['chemical_scaling']['chemical_MSE_multiplier'],16.)
            summary=json.loads((run/'summary.json').read_text());self.assertEqual(summary['chemical_scaling']['output_amplitude'],4.)
            self.assertEqual(summary['chemical_scaling']['residual_normalizer'],1.)
            self.assertEqual(diagnostic['chemical_scaling']['residual_normalizer'],1.)
            self.assertNotEqual((run/'model_initial.eqx').read_bytes(),(run/'model_final.eqx').read_bytes())
            initial=[json.loads((p/'metrics.jsonl').read_text().splitlines()[0])for p in (control,run)]
            for key in ('phi','terminal_phi','velocity','pressure','chemical'):self.assertEqual(initial[0][key],initial[1][key])


if __name__=='__main__':unittest.main()
