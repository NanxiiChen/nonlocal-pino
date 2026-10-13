"""Exact cached support, repeated sampling and capacity checks."""
import unittest
import numpy as np
from scipy.sparse import csr_matrix
from examples.fracture2d.damage_residual_weighting import build_core_halo
from examples.fracture2d.precomputed_damage_sampling import make_sampling_rows,sample_core_halo


class PrecomputedSamplingTest(unittest.TestCase):
    def setUp(self):
        self.D=csr_matrix(np.array([[2.,-1.,0.,-1.],[-1.,1.,0.,0.],[0.,0.,0.,0.],[-1.,0.,0.,1.]]))
        self.rows=make_sampling_rows(self.D,row_width=4)

    def test_all_arrays_match_original_with_repeated_and_empty_rows(self):
        for core in [[3,1,3],[2],[0,1,2,3],[3,0]]:
            old=build_core_halo(self.D,core,node_capacity=4,row_width=4)
            new=sample_core_halo(self.rows,core,node_capacity=4)
            for key in old:np.testing.assert_array_equal(old[key],new[key])
            d=np.array([.2,.4,.8,.9])
            np.testing.assert_allclose(np.sum(new['values']*d[new['global_ids']][new['columns']],axis=1),(self.D@d)[core])
        self.assertFalse(self.rows.columns.flags.writeable)
        self.assertFalse(self.rows.values.flags.writeable)

    def test_capacity_and_inputs_fail_before_truncation(self):
        for core,capacity in [([0],2),([-1],4),([4],4),([.5],4),([],4)]:
            with self.assertRaises(ValueError):sample_core_halo(self.rows,core,node_capacity=capacity)
        with self.assertRaises(ValueError):make_sampling_rows(self.D,row_width=2)


if __name__=='__main__':unittest.main()
