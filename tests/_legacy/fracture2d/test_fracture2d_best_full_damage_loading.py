"""Exact inherited-load identity and restored-state validation for full loading."""
import unittest
import numpy as np
from examples.fracture2d._legacy.configs.best_full_damage_loading import get_config
from examples.fracture2d._legacy.best_full_damage_loading import targets,check_restored_fields
from examples.fracture2d._legacy.configs.full_damage_load_fine import get_config as source_config
from examples.fracture2d._legacy.full_damage_loading import load_grid


class BestFullLoadingTest(unittest.TestCase):
    def test_full_tail_grid_and_user_endpoint(self):
        a=get_config();grid=targets(a)
        self.assertEqual(len(grid),13)
        np.testing.assert_allclose(grid[[0,-1]],[.0053,.0065],rtol=0,atol=1e-14)
        c=source_config();c.full_loading.end_load=a.end_load
        np.testing.assert_array_equal(grid,load_grid(c.full_loading)[52:])
        self.assertEqual(a.max_outer_sweeps,4096)
        a.source_load=.00523
        with self.assertRaises(ValueError):targets(a)

    def test_restart_rejects_wrong_load_or_corrupt_fields(self):
        a=get_config();f=dict(q=np.zeros(6),d=np.array([.1,.2,.3]),
            H=np.ones(3),H_used=np.ones(3),damage_image=np.array([.1,.2,.3]))
        report=dict(index=52,load=.0052,optimizer_count=384000,q_d_synchronized=True)
        check_restored_fields(f,report,a,material_nodes=3,displacement_dofs=6)
        with self.assertRaises(ValueError):
            check_restored_fields(f,dict(report,load=.0053),a,material_nodes=3,displacement_dofs=6)
        with self.assertRaises(ValueError):
            check_restored_fields(dict(f,H=np.full(3,np.nan)),report,a,material_nodes=3,displacement_dofs=6)


if __name__=='__main__':unittest.main()
