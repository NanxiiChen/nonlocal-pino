"""Reject mismatched physics, cold state and actual draw traces in rolling controls."""
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np

from examples.particle_corrosion2d.configs.train_online import get_config
from examples.particle_corrosion2d.online_validation import (
    SHARED_SOURCES, compatible_control_source, validate_control_result, validate_fixed_control)


class ParticleOnlineControlTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.control, self.run = self.root/'fixed', self.root/'rolling'
        self.inputs, self.reference = self.root/'inputs', self.root/'reference'
        for path in (self.control, self.run, self.inputs, self.reference):
            path.mkdir()
        self.c = get_config()
        self.c.train.iterations = 4
        self.c.operator.online.geometry_every = 2
        self.c.operator.online.spatial_every = 1
        prior = json.loads(self.c.to_json())
        prior['operator'].pop('online')
        self.write(self.control/'config.json', prior)
        self.write(self.run/'config.json', json.loads(self.c.to_json()))
        runtime = dict(seed=193, cold_initialization=True,
                       training_uses_reference_targets=False,
                       inputs=str(self.inputs), reference=str(self.reference))
        self.write(self.control/'runtime.json', runtime)
        self.write(self.run/'runtime.json', dict(runtime, sampling_control=str(self.control)))
        for path in (self.control, self.run):
            self.write(path/'summary.json', dict(steps=self.c.train.iterations))
            for name in ('model_initial.eqx', 'optimizer_initial.eqx',
                         'model_final.eqx', 'optimizer_final.eqx'):
                # The validator compares serialized bytes, not a network type.
                (path/name).write_bytes(name.encode()+b'\x00independent fixture\x01')
            for source in SHARED_SOURCES:
                target = path/'source'/source
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(Path(__file__).resolve().parents[1]/source, target)
        rng = np.random.default_rng(runtime['seed'])
        shape = (self.c.train.iterations, self.c.operator.condition_batch)
        self.draws = dict(
            condition_ids=rng.integers(self.c.operator.geometry_count, size=shape, dtype=np.int32),
            pool_ids=rng.integers(self.c.sample.K_pool, size=shape, dtype=np.int32),
            times=rng.uniform(0, self.c.sample.t_max,
                              size=(self.c.train.iterations, self.c.sample.Nt)).astype(np.float32))
        for path in (self.control, self.run):
            np.savez(path/'sampling.npz', **self.draws)

    @staticmethod
    def write(path, value):
        path.write_text(json.dumps(value, allow_nan=False)+'\n')

    def test_accepts_online_only_config_and_actual_cold_trace(self):
        runtime = validate_fixed_control(self.c, self.control, self.inputs, self.reference)
        self.assertEqual(runtime['seed'], 193)
        result = validate_control_result(self.run, self.control)
        self.assertTrue(result['identical_cold_model'])
        self.assertTrue(result['identical_cold_optimizer'])
        self.assertTrue(result['identical_slot_pool_time_draws'])
        saved = json.loads((self.run/'fixed_control_validation.json').read_text())
        self.assertEqual(saved, result)

    def test_sampler_repair_exception_preserves_all_geometry_and_validation_functions(self):
        old, new = self.root/'old_geometry.py', self.root/'new_geometry.py'
        old.write_text('def boundary(x):\n    return x\ndef sample_geometry(c, rng):\n    return rng.random()\n')
        new.write_text('def boundary(x):\n    return x\n'
                       'class GeometrySamplingExhausted(RuntimeError):\n    pass\n'
                       'def sample_geometry(c, rng, *, diagnostics=None):\n'
                       '    value = rng.random()\n    return value\n')
        name = 'examples/particle_corrosion2d/geometry.py'
        self.assertTrue(compatible_control_source(name, old, new))
        self.assertFalse(compatible_control_source('examples/particle_corrosion2d/operators.py', old, new))
        repaired = new.read_text()
        new.write_text(repaired.replace('diagnostics=None', 'relaxed=True'))
        self.assertFalse(compatible_control_source(name, old, new))
        new.write_text(repaired)
        new.write_text(new.read_text().replace('return x', 'return 2*x'))
        self.assertFalse(compatible_control_source(name, old, new))

    def test_rejects_changed_physics_geometry_network_and_optimizer(self):
        changes = (('physical', 'kappa'), ('geometry', 'pit_radius'),
                   ('nl', 'delta'), ('arch', 'width'), ('train', 'lr_init'))
        for group, key in changes:
            with self.subTest(group=group, key=key):
                original = self.c[group][key]
                self.c[group][key] = original*2
                with self.assertRaisesRegex(ValueError, 'only operator.online'):
                    validate_fixed_control(self.c, self.control, self.inputs, self.reference)
                self.c[group][key] = original

    def test_rejects_changed_initial_bank_or_reference_path(self):
        for inputs, reference in ((self.root/'other_inputs', self.reference),
                                  (self.inputs, self.root/'other_reference')):
            with self.subTest(inputs=str(inputs), reference=str(reference)):
                with self.assertRaisesRegex(ValueError, 'initial bank and reference paths'):
                    validate_fixed_control(self.c, self.control, inputs, reference)

    def test_rejects_incomplete_control_and_supervised_provenance(self):
        self.write(self.control/'summary.json', dict(steps=self.c.train.iterations-1))
        with self.assertRaisesRegex(ValueError, 'completed cold physics-only'):
            validate_fixed_control(self.c, self.control, self.inputs, self.reference)
        self.write(self.control/'summary.json', dict(steps=self.c.train.iterations))
        path = self.control/'runtime.json'
        runtime = json.loads(path.read_text())
        runtime['training_uses_reference_targets'] = True
        self.write(path, runtime)
        with self.assertRaisesRegex(ValueError, 'completed cold physics-only'):
            validate_fixed_control(self.c, self.control, self.inputs, self.reference)

    def test_rejects_numerical_source_drift(self):
        source = SHARED_SOURCES[1]
        for directory, message in ((self.control, 'Shared numerical/model source'),
                                   (self.run, 'Actual run source')):
            with self.subTest(directory=directory.name):
                path = directory/'source'/source
                original = path.read_bytes()
                path.write_bytes(original+b'\n# changed numerical source\n')
                with self.assertRaisesRegex(ValueError, message):
                    validate_control_result(self.run, self.control)
                path.write_bytes(original)

    def test_rejects_changed_cold_model_and_optimizer_bytes(self):
        for name in ('model_initial.eqx', 'optimizer_initial.eqx'):
            with self.subTest(name=name):
                path = self.run/name
                original = path.read_bytes()
                path.write_bytes(original[:-1]+b'\x02')
                with self.assertRaisesRegex(ValueError, 'Cold initialization differs'):
                    validate_control_result(self.run, self.control)
                path.write_bytes(original)

    def test_rejects_unfinished_rolling_outputs(self):
        for name in ('model_final.eqx', 'optimizer_final.eqx'):
            with self.subTest(name=name):
                path = self.run/name
                original = path.read_bytes()
                path.unlink()
                with self.assertRaisesRegex(ValueError, 'completed fixed-budget'):
                    validate_control_result(self.run, self.control)
                path.write_bytes(original)

    def test_rejects_changed_actual_slot_pool_and_time_draws(self):
        for name in self.draws:
            with self.subTest(name=name):
                modified = {key: value.copy() for key, value in self.draws.items()}
                modified[name][0, 0] += .01 if name == 'times' else 1
                np.savez(self.run/'sampling.npz', **modified)
                with self.assertRaisesRegex(ValueError, 'Common slot/pool/time draws differ'):
                    validate_control_result(self.run, self.control)
        np.savez(self.run/'sampling.npz', **self.draws)

    def test_rejects_dtype_mismatch_even_when_values_match(self):
        modified = dict(self.draws, times=self.draws['times'].astype(np.float64))
        np.savez(self.run/'sampling.npz', **modified)
        with self.assertRaisesRegex(ValueError, 'Common slot/pool/time draws differ'):
            validate_control_result(self.run, self.control)

    def test_equal_but_truncated_traces_cannot_certify_complete_draws(self):
        truncated = {name: array[:-1] for name, array in self.draws.items()}
        for path in (self.control, self.run):
            np.savez(path/'sampling.npz', **truncated)
        with self.assertRaisesRegex(ValueError, 'shape|budget|length'):
            validate_control_result(self.run, self.control)

    def test_equal_but_invalid_geometry_indices_are_rejected(self):
        invalid = {name: array.copy() for name, array in self.draws.items()}
        invalid['condition_ids'][0, 0] = self.c.operator.geometry_count
        for path in (self.control, self.run):
            np.savez(path/'sampling.npz', **invalid)
        with self.assertRaisesRegex(ValueError, 'range|index|indices|draw'):
            validate_control_result(self.run, self.control)

    def test_equal_but_out_of_interval_time_draws_are_rejected(self):
        invalid = {name: array.copy() for name, array in self.draws.items()}
        invalid['times'][0, 0] = 2*self.c.sample.t_max
        for path in (self.control, self.run):
            np.savez(path/'sampling.npz', **invalid)
        with self.assertRaisesRegex(ValueError, 'time draws outside'):
            validate_control_result(self.run, self.control)


if __name__ == '__main__':
    unittest.main()
