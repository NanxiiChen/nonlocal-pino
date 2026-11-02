"""Strict physics-only auxiliary-head transfer and final provenance audit."""
from pathlib import Path
import json
import shutil
import tempfile
import unittest
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from ml_collections import ConfigDict
from examples.channel_nsch2d.network import build_network,extend_stress_outputs
from examples.channel_nsch2d.refinement import audit_refinement,parent_spec
from examples.channel_nsch2d.stress_initialization import initializer_paths,import_initializer,FILENAMES
from tests.test_channel_stress import configs


class StressInitialization(unittest.TestCase):
    def test_transfer_audit_and_corruptions(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);parent_dir=root/'parent';parent_dir.mkdir();run=root/'run';run.mkdir()
            fit=root/'fit';fit.mkdir();old,c=configs();old.train.seed=42;c.train.seed=42
            old.train.iterations=3;c.train.iterations=3
            parent=build_network(old,jax.random.PRNGKey(13));template=build_network(c,jax.random.PRNGKey(29))
            iso=extend_stress_outputs(parent,template);coef=np.arange(10,dtype=float).reshape(2,5)*.001
            d1,d2=jnp.asarray(coef,dtype=jnp.float32)
            model=eqx.tree_at(lambda m:(m.output.weight,m.output.bias),iso,
                (iso.output.weight.at[3].add(d1[:-1]).at[5].add(-d1[:-1]).at[4].set(d2[:-1]),
                 iso.output.bias.at[3].add(d1[-1]).at[5].add(-d1[-1]).at[4].set(d2[-1])))
            c.train.refinement=ConfigDict(dict(parent_run=str(parent_dir)))
            spec=ConfigDict(dict(kind='physics_auxiliary_block',model=str(fit/'model.eqx'),
                report=str(fit/'fit.json'),quadratic=str(fit/'quadratic.npz')))
            c.train.refinement.stress_initializer=spec
            count=round(c.physical.terminal_time/c.time_discretization.dt)
            nodes=np.arange(1,count+1,dtype=np.float32)*np.float32(c.time_discretization.dt)
            G=np.eye(coef.size);g=-coef.ravel();value=1.-np.sum(coef**2)
            report=dict(status='physics_stress_head_equilibration_passed',parent_run=str(parent_dir),
                parent_config=old.to_dict(),stress_formulation=c.stress_formulation.to_dict(),
                time_nodes=count,input_nodes=81,supervised_evolved_field_samples=0,
                objective_before=1.,objective_after_quadratic=value,objective_after_actual_FP32=value,
                old_GPU_physical_replay_gap=0.,projected_GPU_physical_replay_gap=1e-7,pressure_feature_gap=1e-7)
            (parent_dir/'config.json').write_text(old.to_json());(fit/'fit.json').write_text(json.dumps(report))
            np.savez(fit/'quadratic.npz',coefficients=coef,time=nodes,gram=G,linear=g,constant=1.)
            eqx.tree_serialise_leaves(fit/'model.eqx',model)
            paths=initializer_paths(spec);actual=import_initializer(parent,template,c,paths)
            np.testing.assert_array_equal(actual.output.weight,model.output.weight)
            np.testing.assert_array_equal(actual.output.weight[:3],parent.output.weight[:3])
            np.testing.assert_allclose(-.5*(actual.output.weight[3]+actual.output.weight[5]),parent.output.weight[3],atol=1e-7)
            (parent_dir/'summary.json').write_text(json.dumps(dict(status='completed',iterations=3)))
            (parent_dir/'validation.json').write_text(json.dumps(dict(status='replay_and_hard_constraints_passed')))
            eqx.tree_serialise_leaves(parent_dir/'model_final.eqx',parent)
            for name in ('known_inputs.npz','sampling.npz','prediction.npz','operators.json','optimizer_final.eqx'):
                (parent_dir/name).write_bytes(b'fixture provenance only')
            self.assertEqual(parent_spec(c)[0],parent_dir)
            for src,dst in (('model_final.eqx','model_parent.eqx'),('known_inputs.npz','known_inputs.npz'),
                            ('sampling.npz','sampling.npz'),('optimizer_final.eqx','optimizer_soap.eqx')):
                shutil.copyfile(parent_dir/src,run/dst)
            for name,path in paths.items():shutil.copyfile(path,run/FILENAMES[name])
            eqx.tree_serialise_leaves(run/'model_initial.eqx',actual)
            np.savez(run/'refinement_nodes.npz',time=nodes)
            (run/'refinement_summary.json').write_text(json.dumps(dict(iterations=1,objective_evaluations=1)))
            (run/'refinement_evaluations.jsonl').write_text(json.dumps(dict(evaluation=1,time_nodes=count))+'\n')
            (run/'summary.json').write_text(json.dumps(dict(refinement=dict(iterations=1))))
            c.train.refinement.time_batch=8
            self.assertTrue(audit_refinement(run,c)['auxiliary_stress_initializer_reconstructed'])
            damaged=eqx.tree_at(lambda m:m.output.bias,model,model.output.bias.at[0].add(1.))
            eqx.tree_serialise_leaves(fit/'model.eqx',damaged)
            with self.assertRaises(AssertionError):import_initializer(parent,template,c,paths)
            eqx.tree_serialise_leaves(fit/'model.eqx',model)
            report['parent_config']['physical']['mobility']*=2
            (fit/'fit.json').write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError,'recipe'):import_initializer(parent,template,c,paths)
            with self.assertRaisesRegex(ValueError,'differs'):audit_refinement(run,c)
            spec.model='relative.eqx'
            with self.assertRaisesRegex(ValueError,'absolute'):initializer_paths(spec)


if __name__=='__main__':unittest.main()
