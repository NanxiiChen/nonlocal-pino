"""Saved-field provenance and disjoint timing accounting for sampling reports."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from examples.particle_corrosion2d.compare_sampling import (
    _cohort_table, _costs, _saved_evaluation, compare_sampling)
from examples.particle_corrosion2d.configs.train_online import get_config
from examples.particle_corrosion2d.evaluate import metrics


class ParticleSamplingComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.c = get_config()
        self.c.train.iterations = 3
        self.run = self.root/'run'
        self.reference = self.root/'reference'
        self.evaluation = self.root/'evaluation'
        for path in (self.run, self.reference, self.evaluation):
            path.mkdir()
        self.points = np.array([[.1, .1], [.2, .1], [.1, .2]])
        self.times = np.array([0., .3, 1.])
        self.volumes = np.array([.1, .15, .2])
        phi = np.array([[.2, .7, .9], [.15, .6, .85], [.1, .4, .8]])
        self.target = np.stack((phi, .05+.9*phi), axis=-1)
        records, cases = [], []
        for index in range(8):
            prediction = self.target+.002*(index+1)
            np.savez(self.reference/f'case_{index:03d}.npz', fields=self.target,
                points=self.points, times=self.times, volumes=self.volumes)
            np.savez(self.evaluation/f'prediction_{index:03d}.npz', prediction=prediction,
                points=self.points, times=self.times)
            cases.append(dict(file=f'case_{index:03d}.npz'))
            records.append(metrics(prediction, self.target, self.volumes, self.times))
        self.write(self.reference/'manifest.json', dict(config=json.loads(self.c.to_json()),
            backend='local_p1_lumped_fem', cases=cases))
        result = {name: float(np.mean([row[name] for row in records])) for name in records[0]}
        result['conditions'] = records
        self.write(self.evaluation/'metrics.json', dict(run=str(self.run),
            reference=str(self.reference), checkpoint='model_final.eqx',
            training_performed=False, metrics=result))

    @staticmethod
    def write(path, data):
        path.write_text(json.dumps(data, allow_nan=False)+'\n')

    def test_saved_cohort_recomputed_before_reporting(self):
        checked = _saved_evaluation(self.evaluation, self.run, self.c)
        self.assertEqual(checked['conditions'], 8)
        self.assertGreater(checked['metrics']['phi_pooled'], 0)
        table = _cohort_table(checked, checked)
        self.assertEqual(len(table['table']), 8)
        self.assertEqual(table['table'][3]['fixed'], table['table'][3]['rolling'])

    def test_changed_saved_field_cannot_reuse_old_scalar_metrics(self):
        np.savez(self.evaluation/'prediction_003.npz', prediction=self.target+.2,
                 points=self.points, times=self.times)
        with self.assertRaisesRegex(ValueError, 'recomputed metric differs'):
            _saved_evaluation(self.evaluation, self.run, self.c)

    def test_wrong_model_and_different_reference_path_rejected(self):
        with self.assertRaisesRegex(ValueError, 'this run'):
            _saved_evaluation(self.evaluation, self.root/'different_run', self.c)
        first = _saved_evaluation(self.evaluation, self.run, self.c)
        second = dict(first, reference=str(self.root/'different_reference'))
        with self.assertRaisesRegex(ValueError, 'identical reference path'):
            _cohort_table(first, second)

    def test_background_wait_is_reported_without_double_counting(self):
        self.write(self.run/'timing.json', dict(update_seconds=[4., .02, .03],
            sample_seconds=[.1, .001, .001], evaluation_seconds=[.3, .3],
            setup_seconds=1., total_seconds=7., online_update_seconds=[0., .5, .1]))
        self.write(self.run/'memory.json', [dict(step=1, allocator=None)])
        self.write(self.run/'online_summary.json', dict(producer_wait_seconds=.2,
            capacity=self.c.operator.geometry_count, pde_solves=0, fem_solves=0,
            evolved_training_fields=0))
        result = _costs(self.run, self.c, True)
        self.assertAlmostEqual(result['online_update_total_seconds'], .6)
        self.assertAlmostEqual(result['producer_wait_seconds'], .2)
        self.assertAlmostEqual(result['update_total_seconds'], 4.05)
        timing = json.loads((self.run/'timing.json').read_text())
        timing['online_update_seconds'] = [0., .5]
        self.write(self.run/'timing.json', timing)
        with self.assertRaisesRegex(ValueError, 'completed rolling-production'):
            _costs(self.run, self.c, True)

    def test_one_sided_optional_evaluation_is_rejected_before_output(self):
        output = self.root/'comparison'
        with self.assertRaisesRegex(ValueError, 'Supply both'):
            compare_sampling(self.run, self.run, output, self.evaluation, None)
        self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
