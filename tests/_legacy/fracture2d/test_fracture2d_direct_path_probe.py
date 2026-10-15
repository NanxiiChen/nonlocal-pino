"""Load-step controls and coupled thresholds must be explicit and roundoff-safe."""
import unittest
import numpy as np
from examples.fracture2d._legacy.direct_path_probe import target_grid, contract_limits
from examples.fracture2d._legacy.legacy_coupling_refine import readiness


class DirectPathControlsTest(unittest.TestCase):
    def test_three_partitions_hit_exact_same_endpoint(self):
        start,end=.0061000000000000325,.006200000000000034
        for step,count in ((1e-4,1),(5e-5,2),(2.5e-5,4)):
            grid=target_grid(start,end,step)
            self.assertEqual(len(grid),count)
            self.assertEqual(grid[-1],end)
            self.assertTrue(np.all(np.diff(np.r_[start,grid])>0))
        with self.assertRaisesRegex(ValueError,'partition'):
            target_grid(start,end,3e-5)

    def test_tight_contract_does_not_accept_small_at2_alone(self):
        limits=contract_limits(1e-6)
        diagnostic=dict(at2_relative_inf=1e-8,adjacent_sweep=dict(
            damage_change_inf=2e-6,q_change_relative_inf=1e-8,
            H_change_relative_inf=1e-8,reaction_change_relative=1e-8))
        checks=readiness(diagnostic,dict(physical=True),limits)
        self.assertTrue(checks['coupled_at2'])
        self.assertFalse(checks['damage_change_inf'])
        with self.assertRaises(ValueError):
            contract_limits(np.nan)


if __name__=='__main__':
    unittest.main()
