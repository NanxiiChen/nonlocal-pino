"""Paired-load controls isolate step size and reject invalid endpoint grids."""
import unittest
import numpy as np
from examples.fracture2d._legacy.configs.full_damage_load_coarse import get_config as coarse
from examples.fracture2d._legacy.configs.full_damage_load_fine import get_config as fine
from examples.fracture2d._legacy.full_damage_loading import common_config, validate


class ColdFullLoadingTest(unittest.TestCase):
    def test_paired_settings_and_matching_physical_loads(self):
        c, f = coarse(), fine()
        self.assertEqual(common_config(c), common_config(f))
        cg, fg = validate(c), validate(f)
        self.assertEqual((len(cg), len(fg)), (13, 53))
        np.testing.assert_allclose(cg[-4:], [.0050, .0051, .0052, .0053], rtol=0, atol=1e-14)
        np.testing.assert_allclose(fg[-4:], cg[-4:], rtol=0, atol=1e-14)
        self.assertEqual(round(cg[0]/fg[0]), 5)

    def test_invalid_grid_and_forbidden_constraint_fail(self):
        c=coarse();c.full_loading.pre_step=3e-4
        with self.assertRaises(ValueError):validate(c)
        c=coarse();c.full_loading.symmetry_constraint=True
        with self.assertRaises(ValueError):validate(c)
        c=coarse();c.stagger.omega=.5
        with self.assertRaises(ValueError):validate(c)


if __name__=='__main__':unittest.main()
