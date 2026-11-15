"""Exact boundary membership at floating thresholds and translated geometries."""
import unittest
import numpy as np

from examples.confined_ch2d.operator_indexing import indexed_band
from nonlocal_pinn.training.spatial_sampling import polyline_band


class BoundaryIndexTests(unittest.TestCase):
    def test_original_predicate_threshold_translation_and_random_segments(self):
        rng=np.random.default_rng(613)
        for dtype in (np.float32,np.float64):
            segment=np.array([[[0.,0.],[1.,0.]]],dtype=dtype)
            width=dtype(.125)
            y=np.array([np.nextafter(width,dtype(0)),width,np.nextafter(width,dtype(1))],dtype=dtype)
            points=np.column_stack((np.full(3,.5,dtype=dtype),y))
            np.testing.assert_array_equal(indexed_band(points,segment,width),[True,True,False])
            self.assertEqual(indexed_band(np.empty((0,2),dtype=dtype),segment,width).shape,(0,))
            for scale,translation in ((.001,0.),(1.,0.),(1.,1e5),(100.,-1e4)):
                seg=(rng.normal(size=(29,2,2))*scale+translation).astype(dtype)
                x=(rng.normal(size=(500,2))*scale+translation).astype(dtype)
                for band in (.001*scale,.15*scale,2.*scale):
                    np.testing.assert_array_equal(indexed_band(x,seg,band),polyline_band(x,seg,band))

    def test_empty_candidates_and_invalid_inputs(self):
        seg=np.array([[[0.,0.],[1.,0.]]])
        x=np.array([[10.,10.],[12.,12.]])
        np.testing.assert_array_equal(indexed_band(x,seg,.1),[False,False])
        for points,segments,width in ((x,seg,0.),(x,np.zeros((1,2,2)),.1),
                (x*np.nan,seg,.1),(x,seg.astype(np.float32),.1)):
            with self.assertRaises(ValueError):indexed_band(points,segments,width)


if __name__=='__main__':unittest.main()
