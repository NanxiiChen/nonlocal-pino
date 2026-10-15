"""Interrupted iterates retain accepted ancestry and original numerical controls."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from examples.fracture2d._legacy.configs.linear_damage_relaxed_continue import get_config as previous
from examples.fracture2d._legacy.configs.linear_damage_pending_continue import get_config
from examples.fracture2d._legacy.linear_damage_pending_continue import pending_recovery,validate_recovery_configuration
from examples.fracture2d._legacy.local_damage_trial import sha
from tests._legacy.fracture2d.test_fracture2d_damage_tolerance import mechanics


class PendingRecoveryTest(unittest.TestCase):
    def test_configuration_preserves_equation_budget_and_bounds(self):
        c=get_config();old=dict(config=previous().to_dict())
        validate_recovery_configuration(c,old)
        for group,name,value in [('physical','Gc',2.8),('outer_stop','consecutive_passes',1)]:
            c=get_config();c[group][name]=value
            with self.assertRaisesRegex(ValueError,'differs beyond'):validate_recovery_configuration(c,old)
        c=get_config();c.range.lbfgs.maxiter=16384
        with self.assertRaisesRegex(ValueError,'differs beyond'):validate_recovery_configuration(c,old)
        c=get_config();c.range.lbfgs.negative_tolerance=.002
        with self.assertRaisesRegex(ValueError,'differs beyond'):validate_recovery_configuration(c,old)

    def test_uncommitted_source_rejects_finalization_parent_changes_and_tampering(self):
        m=mechanics()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);folder=root/'state_0053';folder.mkdir()
            q,H=np.zeros(8,np.float64),np.zeros(4,np.float64)
            dp=np.zeros(4,np.float64);d=np.array([-1e-6,.01,.02,.03],np.float64)
            np.savez(root/'accepted_parent_fields.npz',q=q,d=dp,H=H)
            np.savez(folder/'pending_0126_fields.npz',q=q,d=d,H=H)
            (folder/'pending_0126_network.eqx').write_bytes(b'fixture')
            digest=sha(root/'accepted_parent_fields.npz')
            parent=dict(index=52,status='ready',physical_state_promoted=True,
                outer_convergence=dict(converged=True),state_sha256=digest)
            meta=dict(index=53,load=.0053,completed_sweep=126,has_image=True,consecutive=0,
                parent_sha256=digest,fields_file='pending_0126_fields.npz',network_file='pending_0126_network.eqx',
                fields_sha256=sha(folder/'pending_0126_fields.npz'),network_sha256=sha(folder/'pending_0126_network.eqx'),
                reaction=0.,mechanical=dict(convergence_limit=1e-10))
            row=dict(index=53,load=.0053,sweep=126,parent_sha256=digest,outer_convergence=dict(converged=False))
            def write(p,x):p.write_text(json.dumps(x))
            write(root/'accepted_parent_result.json',parent);write(folder/'pending.json',meta);write(folder/'sweep_0126.json',row)
            parents,pending,_,_=pending_recovery(root,m,completed_outer=126,index=53,load=.0053)
            np.testing.assert_array_equal(parents['d'],dp);np.testing.assert_array_equal(pending['d'],d)
            write(root/'result.json',{})
            with self.assertRaisesRegex(ValueError,'non-finalized'):pending_recovery(root,m,completed_outer=126,index=53,load=.0053)
            (root/'result.json').unlink()
            write(folder/'sweep_0126.json',{**row,'outer_convergence':dict(converged=True)})
            with self.assertRaisesRegex(ValueError,'identity'):pending_recovery(root,m,completed_outer=126,index=53,load=.0053)
            write(folder/'sweep_0126.json',row)
            write(root/'accepted_parent_result.json',{**parent,'physical_state_promoted':False})
            with self.assertRaisesRegex(ValueError,'identity'):pending_recovery(root,m,completed_outer=126,index=53,load=.0053)
            write(root/'accepted_parent_result.json',parent)
            (folder/'pending_0126_fields.npz').write_bytes(b'tampered')
            with self.assertRaisesRegex(ValueError,'checksum'):pending_recovery(root,m,completed_outer=126,index=53,load=.0053)
