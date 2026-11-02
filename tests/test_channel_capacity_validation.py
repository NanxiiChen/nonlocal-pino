"""Do not let a finite capacity trial silently change its physical control."""
import json
from pathlib import Path
import tempfile
import unittest

from examples.channel_nsch2d.configs.train_operator_random_walls import get_config as control
from examples.channel_nsch2d.configs.train_operator_random_walls_width192_batch8 import get_config
from examples.channel_nsch2d.capacity_validation import comparison_config, execution_config, audit_probe_state


class CapacityValidationTest(unittest.TestCase):
    def test_only_declared_capacity_and_execution_changes_are_allowed(self):
        old=control();old.train.seed=42
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'config.json').write_text(old.to_json())
            (root/'summary.json').write_text(json.dumps(dict(status='completed')))
            (root/'validation.json').write_text(json.dumps(dict(status='replay_and_hard_constraints_passed')))
            c=get_config();c.train.seed=42
            comparison_config(c,root)
            for change in (lambda c:setattr(c.physical,'mobility',.01),
                           lambda c:setattr(c.arch,'heads',8),
                           lambda c:setattr(c.arch,'spatial_features',192),
                           lambda c:setattr(c.train,'seed',43),
                           lambda c:setattr(c.train.microbatch,'size',8),
                           lambda c:setattr(c.operator,'condition_batch',16)):
                with self.subTest(change=change):
                    candidate=get_config();candidate.train.seed=42;change(candidate)
                    with self.assertRaises(ValueError):comparison_config(candidate,root)

    def test_cold_state_audit_rejects_trained_or_different_optimizer_bytes(self):
        c=get_config();c.train.seed=42
        with tempfile.TemporaryDirectory() as directory:
            run=Path(directory)/'run';probe=Path(directory)/'probe'
            run.mkdir();probe.mkdir()
            for root in (run,probe):
                (root/'config.json').write_text(c.to_json())
                (root/'model_initial.eqx').write_bytes(b'cold-model')
                (root/'optimizer_initial.eqx').write_bytes(b'complete-fresh-SOAP')
            self.assertTrue(audit_probe_state(run,probe)['cold'])
            (run/'optimizer_initial.eqx').write_bytes(b'fitted-SOAP')
            with self.assertRaisesRegex(ValueError,'optimizer_initial'):audit_probe_state(run,probe)

    def test_hardware_execution_allows_only_declared_block_change(self):
        old=control();old.train.seed=42
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'config.json').write_text(old.to_json())
            (root/'summary.json').write_text(json.dumps(dict(status='completed')))
            (root/'validation.json').write_text(json.dumps(dict(status='replay_and_hard_constraints_passed')))
            c=get_config();c.train.seed=42;c.train.microbatch.size=8
            execution_config(c,root)
            with self.assertRaises(ValueError):comparison_config(c,root)
            self.assertEqual(c.train.microbatch.size,8)
            c.train.lr_init=.002
            with self.assertRaises(ValueError):execution_config(c,root)
            c.train.lr_init=.001
            for size in (2,4,5,8,10,20,40):
                c.train.microbatch.size=size;execution_config(c,root)
            c.train.microbatch.size=3
            with self.assertRaisesRegex(ValueError,'40-time divisor'):execution_config(c,root)


if __name__=='__main__':unittest.main()
