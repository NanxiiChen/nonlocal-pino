"""Reject physical/model changes when transferring a frozen geometry operator."""
import unittest
import json
from pathlib import Path
import tempfile
import numpy as np
from examples.channel_nsch2d.configs.train_operator_width128_gmls_time40_jvp_chemical_online import get_config as original
from examples.channel_nsch2d.configs.train_operator_width128_gmls_time40_jvp_stronger_online import get_config
from nonlocal_pinn.training.runtime import with_run_seed
from docs.tasks.channel_geometry_transfer import check_transfer,audit_cold_geometry


class ChannelGeometryTransferTests(unittest.TestCase):
    def test_allow_geometry_metadata_but_reject_changed_physical_field_semantics(self):
        source=with_run_seed(original(),42)
        report=check_transfer(source,with_run_seed(get_config(),42))
        self.assertFalse(report['inside_source_prior'])
        self.assertEqual(report['target_nominal'],[.18,.12])
        for group,name,value in (
                ('physical','mean_velocity',.25),('physical','epsilon',.04),
                ('arch','width',192),('loss','chemical_scale',1.),
                ('train','lr_init',.002),('sample','Nt',80)):
            changed=with_run_seed(get_config(),42)
            setattr(getattr(changed,group),name,value)
            with self.subTest(group=group,name=name),self.assertRaisesRegex(ValueError,'only wall amplitudes'):
                check_transfer(source,changed)

    def test_cold_geometry_audit_checks_state_and_time_stream_without_equating_geometries(self):
        with tempfile.TemporaryDirectory()as directory:
            source,target=(Path(directory)/name for name in ('source','target'))
            for p,factory in ((source,original),(target,get_config)):
                p.mkdir();c=with_run_seed(factory(),42)
                c.train.iterations=2;c.operator.online.refresh_every=1
                (p/'config.json').write_text(c.to_json())
                (p/'validation.json').write_text(json.dumps(dict(status='replay_and_hard_constraints_passed')))
                (p/'online-validation.json').write_text(json.dumps(dict(status='actual_rolling_inputs_reconstructed')))
                for name in ('model_initial.eqx','optimizer_initial.eqx'):(p/name).write_bytes(b'paired-state-fixture')
                amplitudes=np.tile(np.array([[c.geometry.upper,c.geometry.lower]],dtype=np.float32),(16,1))
                for name in ('known_inputs.npz','known_inputs_final.npz'):np.savez(p/name,amplitudes=amplitudes)
                times=np.arange(80,dtype=np.float32).reshape(2,40)/80*6
                np.savez(p/'sampling.npz',indices=np.zeros((2,2),dtype=np.int32),times=times,
                    draw_ids=np.arange(4).reshape(2,2),amplitudes=np.tile(amplitudes[0],(2,2,1)))
            result=audit_cold_geometry(target,source)
            self.assertTrue(result['geometry_arrays_intentionally_different'])
            self.assertTrue(result['actual_times_indices_and_online_draw_ids_exact'])
            (target/'optimizer_initial.eqx').write_bytes(b'changed-state')
            with self.assertRaisesRegex(ValueError,'optimizer_initial'):audit_cold_geometry(target,source)
            (target/'optimizer_initial.eqx').write_bytes(b'paired-state-fixture')
            with np.load(target/'sampling.npz')as f:draws={k:f[k]for k in f.files}
            draws['times'][0,0]+=.1;np.savez(target/'sampling.npz',**draws)
            with self.assertRaises(AssertionError):audit_cold_geometry(target,source)


if __name__=='__main__':unittest.main()
