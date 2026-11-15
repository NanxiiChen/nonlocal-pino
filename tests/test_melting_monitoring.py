"""Verify full-test opt-in without changing legacy selection or config numerics."""
import unittest

from examples.icemelting3d.configs.train_operator import get_config as production
from examples.icemelting3d.configs.train_operator_compare_nonlocal import get_config as nonlocal_config
from examples.icemelting3d.configs.train_operator_compare_local_ad import get_config as ad_config
from examples.icemelting3d.configs.train_operator_compare_fem import get_config as fem_config
from examples.icemelting3d.spatial_training import monitoring_spec


class MonitoringTests(unittest.TestCase):
    def test_complete_test_cohort_and_legacy_selection(self):
        manifest = dict(monitor_indices=[0, 1], test_indices=list(range(2, 66)))
        for factory in (nonlocal_config, ad_config, fem_config):
            c = factory()
            before = c.to_json()
            record = monitoring_spec(c, manifest)
            self.assertEqual(record['indices'], manifest['test_indices'])
            self.assertEqual(record['split'], 'test')
            self.assertEqual(record['count'], 64)
            self.assertFalse(record['training_labels'])
            self.assertEqual(c.to_json(), before)
        c = production()
        self.assertNotIn('monitor_split', c.operator)
        self.assertEqual(monitoring_spec(c, manifest)['indices'], [0, 1])

    def test_invalid_split_and_indices_fail_before_training(self):
        c = nonlocal_config()
        for ids in ([], [2, 2], [-1, 2], [2.5, 3.], [True, False], [[2, 3]]):
            with self.assertRaises(ValueError):
                monitoring_spec(c, dict(test_indices=ids))
        c.operator.monitor_split = 'validation'
        with self.assertRaisesRegex(ValueError, 'test or monitor'):
            monitoring_spec(c, dict(test_indices=[2]))


if __name__ == '__main__':
    unittest.main()
