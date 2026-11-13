"""Independent checks for regional norm partition and reduction order."""
import unittest
import numpy as np
from docs.tasks.pino_regional_metrics import reduce_regions, assert_norm_replay


class RegionalMetricsTests(unittest.TestCase):
    def test_large_partition_norm_replay_and_corruption_detection(self):
        rng=np.random.default_rng(52)
        target=rng.uniform(.01,1.,size=(21,10003,2))
        prediction=target+rng.normal(0,.03,size=target.shape)
        near=np.arange(target.shape[1])%7==0
        result=reduce_regions(prediction,target,dict(near=near,bulk=~near),
                              np.ones(target.shape[:2]))
        extended=np.sqrt(np.sum((prediction-target)**2,axis=(0,1),dtype=np.longdouble)
                         /np.sum(target**2,axis=(0,1),dtype=np.longdouble))
        assert_norm_replay(result['all']['channel_errors'],extended)
        with self.assertRaises(AssertionError):
            assert_norm_replay(result['all']['channel_errors'],extended*1.000001)
        # Regression: different valid summation orders must not reject a norm.
        assert_norm_replay([.054671668468492404],[.05467166846850347])

    def test_region_energy_and_channel_reduction(self):
        target=np.ones((3,4,2));target[...,1]=2
        prediction=target.copy();prediction[:,0,0]+=1;prediction[:,0,1]+=1
        near=np.array([True,False,False,False])
        got=reduce_regions(prediction,target,dict(near=near,bulk=~near),np.ones((3,4)))
        np.testing.assert_allclose(got['all']['channel_errors'],[.5,.25])
        self.assertEqual(got['all']['channel_mean'],.375)
        np.testing.assert_array_equal(got['near']['error_energy_fraction'],[1.,1.])
        np.testing.assert_array_equal(got['bulk']['channel_errors'],[0.,0.])
        self.assertEqual(got['near']['channel_mean'],.75)

    def test_physical_measure_matches_independent_integral(self):
        target=np.ones((3,4,2));prediction=target.copy();prediction[:,0]+=2
        weights=np.outer([.1,.5,.4],[1.,2.,3.,4.])
        near=np.array([True,False,False,False])
        got=reduce_regions(prediction,target,dict(near=near,bulk=~near),weights)
        np.testing.assert_allclose(got['all']['channel_errors'],np.sqrt(.4))
        np.testing.assert_allclose(got['all']['rmse'],np.sqrt(.4))

    def test_overlap_and_missing_points_are_rejected(self):
        fields=np.ones((2,3,2));weights=np.ones((2,3))
        for masks in (dict(a=np.array([True,True,False]),b=np.array([False,True,True])),
                      dict(a=np.array([True,False,False]),b=np.array([False,True,False]))):
            with self.assertRaisesRegex(ValueError,'partition'):
                reduce_regions(fields,fields,masks,weights)


if __name__=='__main__':unittest.main()
