"""Restart invariants: preceding-load floor, persistent count and pool boundary."""
import unittest

from examples.fracture2d._legacy.configs.staggered_damage_644 import get_config
from examples.fracture2d._legacy.configs.staggered_damage_644_inner128 import get_config as reduced_config
from examples.fracture2d._legacy.staggered_damage_continue import restart_plan


class RestartTest(unittest.TestCase):
    def test_equal_outer_counts_preserve_distinct_adam_work(self):
        baseline, reduced = get_config(), reduced_config()
        first = restart_plan(baseline, 512, 256)
        second = restart_plan(reduced, 512, 256)
        self.assertEqual(first['initial_count'], 851968)
        self.assertEqual(first['initial_count']+first['updates'], 786432+644*512)
        self.assertEqual(second['initial_count'], 786432)
        self.assertEqual(second['initial_count']+second['updates'], 786432+644*128)
        self.assertEqual(baseline.parent_load, .0052)
        self.assertEqual(reduced.parent_load, baseline.parent_load)
        self.assertEqual(reduced.pins, baseline.pins)

    def test_reject_changed_continuation_budget_and_partial_pool(self):
        config = get_config()
        config.inner_updates = 128
        with self.assertRaises(ValueError): restart_plan(config, 512, 256)
        config = reduced_config()
        config.cases = [dict(name='pinn_full', solver='pinn', omega=1., max_sweeps=643)]
        with self.assertRaises(ValueError): restart_plan(config, 512, 256)
        config = reduced_config()
        config.restart_root = 'unexpected-current-load-parent'
        with self.assertRaises(ValueError): restart_plan(config, 512, 256)


if __name__ == '__main__':
    unittest.main()
