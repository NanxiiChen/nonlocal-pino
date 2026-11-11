"""Rolling method comparison rejects unequal physical inputs and cold states."""
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np

from examples.particle_corrosion2d.configs.train_physical_spaced_w256_method_nonlocal import get_config
from examples.particle_corrosion2d.configs.train_physical_spaced_w256_method_local_ad import (
    get_config as ad_config,
)
from examples.particle_corrosion2d.method_validation import (
    CoreTrace, SOURCES, audit_pair, core_shape, validate_config, validate_control,
)
from examples.particle_corrosion2d.problem import validate
from examples.particle_corrosion2d.online import validate_online
from examples.particle_corrosion2d.strategies import validate_strategies


def write(path, value):
    path.write_text(json.dumps(value, allow_nan=False)+'\n')


class RollingMethodTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.c = get_config()
        self.c.train.iterations = 9
        self.c.operator.geometry_count = 2
        self.c.operator.condition_batch = 1
        self.c.sample.K_core = 2
        self.c.sample.K_pool = 2
        self.c.operator.online.geometry_every = 4
        self.c.operator.online.spatial_every = 2
        self.c.operator.online.geometry_size = 1
        self.c.operator.online.spatial_size = 1
        self.inputs, self.reference = self.root/'inputs', self.root/'reference'
        self.inputs.mkdir(); self.reference.mkdir()
        self.control, self.run = self.root/'nonlocal', self.root/'ad'
        self.bank = dict(points=np.array([[.2, .3], [.4, .5]], dtype=np.float32))
        self.exclusions = [dict(particles=[dict(center=[.3, .4])])]
        self.sources = Path(__file__).resolve().parents[1]
        self._make_run(self.control, self.c, None)
        self.ad = self.c.copy_and_resolve_references()
        self.ad.nl.method = 'local_ad'
        self.ad.loss.gradnorm_bias = tuple(self.ad.loss.gradnorm_bias)+(1., 1.)
        self._make_run(self.run, self.ad, self.control)

    def _make_run(self, run, c, control):
        run.mkdir()
        write(run/'config.json', json.loads(c.to_json()))
        write(run/'runtime.json', dict(seed=42, inputs=str(self.inputs), reference=str(self.reference),
            rolling_inputs=True, cold_initialization=True, training_uses_reference_targets=False,
            rolling_method_control=None if control is None else str(control),
            excluded_reference_paths=[], paired_optimizer_initialization=control is not None))
        write(run/'summary.json', dict(steps=c.train.iterations))
        write(run/'online_validation.json', dict(status='passed', run=str(run), updates=c.train.iterations,
            evaluation_overlap=0, future_training_fields=0, exact_final_slots_reconstructed=2,
            unchanged_initial_slots=0))
        for name in ('model_initial.eqx', 'optimizer_initial.eqx', 'model_final.eqx', 'optimizer_final.eqx'):
            (run/name).write_bytes(b'known identical bytes: '+name.encode())
        for name in SOURCES:
            target = run/'source'/name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(self.sources/name, target)
        for name in ('online_initial_inputs.npz', 'online_final_inputs.npz'):
            np.savez(run/name, **self.bank)
        write(run/'excluded_geometries.json', self.exclusions)
        from examples.particle_corrosion2d.sampling import sample_times
        rng = np.random.default_rng(42)
        np.savez(run/'sampling.npz', condition_ids=rng.integers(2, size=(9, 1), dtype=np.int32),
            pool_ids=rng.integers(2, size=(9, 1), dtype=np.int32), times=sample_times(rng, c))
        np.savez(run/'online_sampling.npz', geometry_draw_ids=np.zeros((9, 1), dtype=np.int64),
            spatial_versions=np.zeros((9, 1), dtype=np.int64))
        trace = CoreTrace(run, c)
        for i in range(c.train.iterations):
            trace.record(i, self.bank['points'][None], np.ones((1, 2), dtype=np.float32))
        trace.finish()
        journal = dict(event=0, completed_updates=2, slots=[0], draw_ids=[-1], spatial_versions=[1],
            wait_seconds=.1, records=[dict(slot=0, draw_id=-1, spatial_version=1,
                geometry=self.exclusions[0], worker_seconds=.2, cache_hit=False)])
        write(run/'online_refresh.jsonl', journal)
        write(run/f'online_pool_{c.train.iterations:06d}.json', dict(completed_updates=c.train.iterations))

    def test_current_selected_configs_validate_without_altering_default(self):
        from examples.particle_corrosion2d.configs.train_physical_spaced_w256 import get_config as selected
        original = selected()
        for c in (get_config(), ad_config()):
            for check in (validate, validate_online, validate_strategies, validate_config):
                check(c)
            self.assertEqual(c.physical, original.physical)
            self.assertEqual(c.arch, original.arch)
            self.assertEqual(c.train, original.train)
            self.assertEqual(c.log.checkpoint_every, 5000)
        self.assertNotIn('method_comparison', original.operator)

    def test_actual_pair_and_worker_scheduling_variations_pass(self):
        row = json.loads((self.run/'online_refresh.jsonl').read_text())
        row['wait_seconds'] = 10.
        row['records'][0]['cache_hit'] = True
        row['records'][0]['worker_seconds'] = 4.
        write(self.run/'online_refresh.jsonl', row)
        result = audit_pair(self.run, self.control)
        self.assertTrue(result['identical_all_geometry_spatial_core_time_draws'])
        self.assertEqual(result['actual_core_queries'], 18)

    def test_method_selector_does_not_change_rolling_physical_inputs(self):
        from examples.particle_corrosion2d.online import build_input
        from tests.test_particle_corrosion_online import small_config
        c = small_config()
        first, first_meta = build_input(c.to_dict(), 314159, 0, 0, None)
        c.nl.method = 'local_ad'
        c.loss.gradnorm_bias = tuple(c.loss.gradnorm_bias)+(1., 1.)
        second, second_meta = build_input(c.to_dict(), 314159, 0, 0, None)
        self.assertEqual(first_meta['geometry'], second_meta['geometry'])
        for name in first:
            np.testing.assert_array_equal(first[name], second[name], err_msg=name)

    def test_method_costs_separate_graph_wait_trace_and_gradient_updates(self):
        from examples.particle_corrosion2d.compare import run_costs
        write(self.run/'timing.json', dict(update_seconds=[.2]*9, sample_seconds=[.01]*9,
            evaluation_seconds=[.1, .2], setup_seconds=1., total_seconds=4.,
            online_update_seconds=[.02]*9, core_trace_seconds=[.001]*9,
            core_trace_finish_seconds=.05))
        write(self.run/'memory.json', [dict(step=1, allocator=None)])
        write(self.run/'online_summary.json', dict(producer_wait_seconds=.03, installed_worker_seconds=2.))
        costs = run_costs(self.run, self.ad)
        self.assertAlmostEqual(costs['update_total_seconds'], 1.8)
        self.assertAlmostEqual(costs['core_trace_total_seconds'], .059)
        self.assertAlmostEqual(costs['online_update_total_seconds'], .18)
        self.assertEqual(costs['installed_worker_cpu_seconds'], 2.)
        self.assertIn('unused graph preparation', costs['paired_graph_overhead'])

    def test_configuration_and_bank_changes_rejected_before_training(self):
        bad = self.ad.copy_and_resolve_references()
        bad.physical.L *= 2
        with self.assertRaisesRegex(ValueError, 'only method'):
            validate_control(bad, self.control, self.inputs, self.reference, self.bank, self.exclusions)
        bad = self.ad.copy_and_resolve_references()
        bad.loss.gradnorm_bias = (1.,)*6+(2.,)
        with self.assertRaisesRegex(ValueError, 'unit Neumann'):
            validate_control(bad, self.control, self.inputs, self.reference, self.bank, self.exclusions)
        changed = dict(points=self.bank['points']+.01)
        with self.assertRaisesRegex(ValueError, 'Initial rolling bank'):
            validate_control(self.ad, self.control, self.inputs, self.reference, changed, self.exclusions)

    def test_changed_actual_core_trace_is_not_hidden_by_matching_seed(self):
        with np.load(self.run/'core_trace.npz') as arrays:
            cores = arrays['cores']
        cores[4, 0, 1, 0] += .01
        np.savez(self.run/'core_trace.npz', cores=cores)
        with self.assertRaisesRegex(ValueError, 'core_trace.npz'):
            audit_pair(self.run, self.control)

    def test_changed_geometry_or_time_draw_cannot_reuse_old_receipt(self):
        row = json.loads((self.run/'online_refresh.jsonl').read_text())
        row['records'][0]['geometry']['particles'][0]['center'][0] += .1
        write(self.run/'online_refresh.jsonl', row)
        with self.assertRaisesRegex(ValueError, 'installed geometry'):
            audit_pair(self.run, self.control)
        shutil.copyfile(self.control/'online_refresh.jsonl', self.run/'online_refresh.jsonl')
        with np.load(self.run/'sampling.npz') as arrays:
            draws = {name: arrays[name] for name in arrays.files}
        draws['times'][2, 0] += .01
        np.savez(self.run/'sampling.npz', **draws)
        with self.assertRaisesRegex(ValueError, 'sampling replay'):
            audit_pair(self.run, self.control)

    def test_identical_but_corrupt_sampling_cannot_pass_by_agreement(self):
        with np.load(self.run/'sampling.npz') as arrays:
            draws = {name: arrays[name] for name in arrays.files}
        draws['condition_ids'][0, 0] = self.c.operator.geometry_count
        for run in (self.run, self.control):
            np.savez(run/'sampling.npz', **draws)
        with self.assertRaisesRegex(ValueError, 'sampling replay'):
            audit_pair(self.run, self.control)

    def test_cold_optimizer_source_and_missing_audit_are_rejected(self):
        (self.run/'optimizer_initial.eqx').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'Cold rolling method state'):
            audit_pair(self.run, self.control)
        shutil.copyfile(self.control/'optimizer_initial.eqx', self.run/'optimizer_initial.eqx')
        (self.run/'source'/SOURCES[0]).write_text('# altered model\n')
        with self.assertRaisesRegex(ValueError, 'source differs'):
            audit_pair(self.run, self.control)
        shutil.copyfile(self.control/'source'/SOURCES[0], self.run/'source'/SOURCES[0])
        receipt = json.loads((self.run/'online_validation.json').read_text())
        receipt['future_training_fields'] = 1
        write(self.run/'online_validation.json', receipt)
        with self.assertRaisesRegex(ValueError, 'independent online'):
            audit_pair(self.run, self.control)

    def test_trace_order_completeness_and_publication(self):
        other = self.root/'trace'
        other.mkdir()
        trace = CoreTrace(other, self.c)
        points = self.bank['points'][None]
        weights = np.ones((1, 2), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, 'ordered'):
            trace.record(1, points, weights)
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            trace.finish()
        with self.assertRaisesRegex(ValueError, 'positive'):
            trace.record(0, points, weights*0)
        for i in range(self.c.train.iterations):
            trace.record(i, points, weights)
        report = trace.finish()
        self.assertEqual(report['shape'], list(core_shape(self.c)))
        self.assertFalse((other/'core_trace_incomplete.npy').exists())


if __name__ == '__main__':
    unittest.main()
