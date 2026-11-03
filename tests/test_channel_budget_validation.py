"""Guard the budget/LR comparison against changed physics or imported state."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from examples.channel_nsch2d.budget_validation import comparison_config, resource_evidence, audit_cold, audit_completed
from examples.channel_nsch2d.configs.train_operator_random_walls_width192_batch8_time20 import get_config as baseline
from examples.channel_nsch2d.configs.train_operator_random_walls_width192_batch8_time20_cold20k import get_config


class BudgetValidationTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.control = self.root/'control'
        self.control.mkdir()
        old = baseline()
        old.train.seed = 42
        (self.control/'config.json').write_text(old.to_json())
        for name, status in (('summary.json', 'completed'),
                             ('validation.json', 'replay_and_hard_constraints_passed'),
                             ('online-validation.json', 'actual_rolling_inputs_reconstructed')):
            (self.control/name).write_text(json.dumps(dict(status=status)))
        for name in ('model_initial.eqx', 'optimizer_initial.eqx'):
            (self.control/name).write_bytes(('unfitted-'+name).encode())
        self.c = get_config()
        self.c.train.seed = 42

    def test_only_budget_decay_and_name_can_change(self):
        comparison_config(self.c, self.control)
        for change in (lambda c: setattr(c.physical, 'mobility', .01),
                       lambda c: setattr(c.train, 'seed', 43),
                       lambda c: setattr(c.train, 'lr_init', .002),
                       lambda c: setattr(c.train, 'lr_min', 1e-6),
                       lambda c: setattr(c.train.microbatch, 'size', 8),
                       lambda c: setattr(c.operator, 'condition_batch', 16),
                       lambda c: setattr(c.arch, 'width', 256)):
            candidate = get_config()
            candidate.train.seed = 42
            change(candidate)
            with self.subTest(change=change), self.assertRaises(ValueError):
                comparison_config(candidate, self.control)

    def test_existing_optimizer_schedule_reaches_the_declared_endpoint(self):
        import optax
        schedule = optax.exponential_decay(self.c.train.lr_init,
            self.c.train.lr_decay_every, self.c.train.lr_decay_rate, end_value=self.c.train.lr_min)
        np.testing.assert_allclose([schedule(0), schedule(10000), schedule(20000)],
                                   [.001, .0001, .00001], rtol=2e-6)
        self.assertLess(float(schedule(19999)), .00001001)
        self.assertGreater(.001*.7**5, float(schedule(10000)))

    def test_smaller_execution_block_requires_explicit_narrow_declaration(self):
        from examples.channel_nsch2d.configs.train_operator_random_walls_width192_batch8_time8_cold20k import get_config as memory_config
        candidate = memory_config()
        candidate.train.seed = 42
        with self.assertRaises(ValueError):
            comparison_config(candidate, self.control)
        comparison_config(candidate, self.control, execution_time_block=8)
        candidate.sample.Nt = 32
        with self.assertRaises(ValueError):
            comparison_config(candidate, self.control, execution_time_block=8)
        with self.assertRaises(ValueError):
            comparison_config(self.c, self.control, execution_time_block=10)

    def test_cold_state_and_known_input_changes_are_rejected(self):
        run = self.root/'run'
        run.mkdir()
        (run/'config.json').write_text(self.c.to_json())
        for name in ('model_initial.eqx', 'optimizer_initial.eqx'):
            (run/name).write_bytes((self.control/name).read_bytes())
        for name in ('known_inputs.npz', 'evaluation_inputs.npz', 'reserved_geometries.npz'):
            for path in (run, self.control):
                np.savez(path/name, values=np.arange(4))
        self.assertTrue(audit_cold(run, self.control)['model_and_full_optimizer_exact'])
        (run/'optimizer_initial.eqx').write_bytes(b'fitted-state')
        with self.assertRaisesRegex(ValueError, 'optimizer_initial'):
            audit_cold(run, self.control)
        (run/'optimizer_initial.eqx').write_bytes((self.control/'optimizer_initial.eqx').read_bytes())
        np.savez(run/'known_inputs.npz', values=np.arange(4)+1)
        with self.assertRaises(AssertionError):
            audit_cold(run, self.control)

    def test_declared36_time_sampling_requires_independent_replay(self):
        from examples.channel_nsch2d.configs.train_operator_random_walls_width192_batch8_time18_nt36_cold20k import get_config as time36
        c = time36(); c.train.seed = 42
        with self.assertRaises(ValueError):
            comparison_config(c, self.control)
        comparison_config(c, self.control, execution_time_block=18)
        wrong = time36(); wrong.train.seed = 42; wrong.sample.Nt = 40
        with self.assertRaises(ValueError):
            comparison_config(wrong, self.control, execution_time_block=18)
        run = self.root/'time36'; run.mkdir()
        (run/'config.json').write_text(c.to_json())
        for name in ('model_initial.eqx', 'optimizer_initial.eqx'):
            (run/name).write_bytes((self.control/name).read_bytes())
        for name in ('known_inputs.npz', 'evaluation_inputs.npz', 'reserved_geometries.npz'):
            for directory in (run, self.control):
                np.savez(directory/name, values=np.arange(4))
        (run/'summary.json').write_text(json.dumps(dict(iterations=20000)))
        (run/'online-validation.json').write_text(json.dumps(dict(status='actual_rolling_inputs_reconstructed')))
        report=dict(status='replay_and_hard_constraints_passed',
                    actual_draws_reproduced_from_global_keys=False,
                    actual_time_strata_passed=True,time_derivative=dict(actual_time_samples=36))
        (run/'validation.json').write_text(json.dumps(report))
        np.savez(self.control/'sampling.npz',indices=np.zeros((10000,8),int),times=np.zeros((10000,40)))
        np.savez(run/'sampling.npz',indices=np.zeros((20000,8),int),times=np.zeros((20000,36)))
        with self.assertRaisesRegex(ValueError,'independent'):
            audit_completed(run,self.control,execution_time_block=18)
        report['actual_draws_reproduced_from_global_keys']=True
        (run/'validation.json').write_text(json.dumps(report))
        result=audit_completed(run,self.control,execution_time_block=18)
        self.assertTrue(result['first10000_geometry_slots_draws_and_amplitudes_exact'])
        self.assertFalse(result['first10000_geometry_slots_draws_amplitudes_and_times_exact'])
        changed=np.zeros((20000,8),int);changed[0,0]=1
        np.savez(run/'sampling.npz',indices=changed,times=np.zeros((20000,36)))
        with self.assertRaises(AssertionError):
            audit_completed(run,self.control,execution_time_block=18)

    def test_resource_reuse_requires_the_same_gpu_and_execution(self):
        probe = self.root/'probe'
        probe.mkdir()
        old = json.loads((self.control/'config.json').read_text())
        gpu = dict(name='qualified-GPU', uuid='same-device', total_memory_bytes=1000)
        result = dict(config=old, status='finite_production_probe_passed', updates=11,
            hardware_trial=dict(gpu_name=gpu['name'], gpu_uuid=gpu['uuid'], total_memory_bytes=1000,
                                allocator_fraction=.97),
            memory=dict(peak_bytes_in_use=900, bytes_limit=970),
            fitted_probe_weights_exported=False, warm_update_seconds=1.88,
            capacity_trial=dict(full_geometry_pool_on_device=True, finite_nominal_evaluation=True))
        (probe/'result.json').write_text(json.dumps(result))
        for name in ('model_initial.eqx', 'optimizer_initial.eqx'):
            (probe/name).write_bytes((self.control/name).read_bytes())
        resource_evidence(self.c, self.control, probe, gpu, .97)
        with self.assertRaises(ValueError):
            resource_evidence(self.c, self.control, probe, dict(gpu, uuid='different-device'), .97)
        with self.assertRaises(ValueError):
            resource_evidence(self.c, self.control, probe, gpu, .95)


if __name__ == '__main__':
    unittest.main()
