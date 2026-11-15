"""Independent weighted fits and whole-trajectory empirical bootstrap checks."""
from itertools import product
import unittest
import numpy as np
from scripts.trajectory_trend import trajectory_interval


class TrajectoryTrendTests(unittest.TestCase):
    def test_two_case_exact_band_and_independent_local_least_squares(self):
        x=np.arange(21,dtype=float);base=np.exp(-.05*x)*(1+.03*np.sin(x))
        values=np.column_stack((base,2.5*base));result=trajectory_interval(x,values,
            fraction=.3,replicates=2000,seed=12,grid_points=61)
        np.testing.assert_array_equal(result['bootstrap_geometry_indices'],list(product(range(2),repeat=2)))
        means=np.column_stack((base,1.75*base,1.75*base,2.5*base))
        estimates=[]
        for point in result['x']:
            distance=x-point;radius=np.sort(abs(distance))[5]
            w=(1-np.minimum(abs(distance/radius),1)**3)**3
            design=np.column_stack((np.ones(len(x)),distance))
            # Independent weighted least squares, without production smoother matrix.
            coefs=np.linalg.lstsq(design*np.sqrt(w)[:,None],np.log10(means)*np.sqrt(w)[:,None],rcond=None)[0]
            estimates.append(10**coefs[0])
        low,high=np.quantile(np.asarray(estimates),[.025,.975],axis=1)
        np.testing.assert_allclose(result['lower'],low,rtol=3e-12)
        np.testing.assert_allclose(result['upper'],high,rtol=3e-12)
        np.testing.assert_allclose(result['raw_mean'],values.mean(axis=1),rtol=0,atol=0)

    def test_eight_case_indices_preserve_temporal_pairing_and_scale(self):
        x=np.arange(101,dtype=float);values=(1+x[:,None]/100)*np.arange(1.,9.)
        kw=dict(fraction=.22,replicates=2000,seed=20261114,grid_points=401)
        a=trajectory_interval(x,values,**kw);b=trajectory_interval(x,values*3,**kw)
        self.assertEqual(a['bootstrap_geometry_indices'].shape,(2000,8))
        np.testing.assert_array_equal(a['bootstrap_geometry_indices'],b['bootstrap_geometry_indices'])
        for key in ('raw_mean','smooth','lower','upper'):np.testing.assert_allclose(b[key],3*a[key],rtol=2e-13)
        for bad in (values[:,:1],-values,values.copy()*np.nan):
            with self.assertRaises(ValueError):trajectory_interval(x,bad,**kw)


if __name__=='__main__':unittest.main()
