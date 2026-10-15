"""Additional training must retain accuracy and an accepted physical parent."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from examples.fracture2d._legacy.configs.linear_damage_continue import get_config as previous
from examples.fracture2d._legacy.configs.linear_damage_budget_continue import get_config
from examples.fracture2d._legacy.linear_damage_budget_continue import recovery_fields, validate_recovery_configuration
from examples.fracture2d._legacy.local_damage_trial import sha
from tests._legacy.fracture2d.test_fracture2d_damage_tolerance import mechanics


class BudgetRecoveryTest(unittest.TestCase):
    def test_config_rejects_equation_or_model_relaxation(self):
        c = get_config()
        original = dict(config=previous().to_dict())
        validate_recovery_configuration(c,original)
        self.assertEqual(c.range.lbfgs.maxiter,8192)
        self.assertEqual(c.range.lbfgs.residual_limit,.001)
        self.assertEqual(c.reporting.range_atol,2e-6)
        for group,name,value in [('range','method','residual_l2'),('physical','Gc',2.8),
                                 ('reporting','coupled_AT2_relative_limit',.01),
                                 ('outer_stop','consecutive_passes',1)]:
            altered = get_config()
            altered[group][name] = value
            with self.assertRaisesRegex(ValueError,'differs beyond'):
                validate_recovery_configuration(altered,original)
        altered = get_config();altered.range.lbfgs.residual_limit=.002
        with self.assertRaisesRegex(ValueError,'differs beyond'):
            validate_recovery_configuration(altered,original)

    def test_recovery_retains_parent_and_rejects_promotion_or_tampering(self):
        m = mechanics()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);failed=root/'state_0053';failed.mkdir()
            q,H=np.zeros(8,np.float64),np.zeros(4,np.float64)
            dp=np.zeros(4,np.float64);d=np.array([-4.4e-7,.01,.02,.03],np.float64)
            np.savez(root/'accepted_parent_fields.npz',q=q,d=dp,H=H)
            np.savez(failed/'pending_0123_fields.npz',q=q,d=d,H=H)
            np.savez(failed/'fields.npz',q=q,d=d,H=H,H_used=H,d_parent=dp,H_parent=H)
            for p in (failed/'network.eqx',failed/'pending_0123_network.eqx'):
                p.write_bytes(b'checksum-fixture')
            parent_sha=sha(root/'accepted_parent_fields.npz')
            parent=dict(status='ready',physical_state_promoted=True,outer_convergence=dict(converged=True),state_sha256=parent_sha)
            fr=dict(status='inner_rejected',physical_state_promoted=False,inner_calls=124,
                accepted_inner_sweeps=123,parent_sha256=parent_sha,state_sha256=sha(failed/'fields.npz'),
                network_sha256=sha(failed/'network.eqx'))
            meta=dict(completed_sweep=123,index=53,load=.0053,has_image=True,parent_sha256=parent_sha,
                fields_file='pending_0123_fields.npz',network_file='pending_0123_network.eqx',
                fields_sha256=sha(failed/'pending_0123_fields.npz'),network_sha256=sha(failed/'pending_0123_network.eqx'),
                reaction=0.,mechanical=dict(convergence_limit=1e-10))
            def write(p,x):p.write_text(json.dumps(x))
            write(root/'accepted_parent_result.json',parent);write(failed/'result.json',fr);write(failed/'pending.json',meta)
            parents,pending,_=recovery_fields(root,m,parent_index=52,completed_outer=123,load=.0053)
            np.testing.assert_array_equal(parents['d'],dp);np.testing.assert_array_equal(pending['d'],d)
            write(failed/'result.json',{**fr,'physical_state_promoted':True})
            with self.assertRaisesRegex(ValueError,'ancestry'):
                recovery_fields(root,m,parent_index=52,completed_outer=123,load=.0053)
            write(failed/'result.json',fr)
            (failed/'pending_0123_fields.npz').write_bytes(b'tampered')
            with self.assertRaisesRegex(ValueError,'checksum'):
                recovery_fields(root,m,parent_index=52,completed_outer=123,load=.0053)
