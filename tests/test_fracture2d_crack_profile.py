"""Path diagnostics distinguish straight, bent and undeveloped synthetic fields."""
import unittest
import numpy as np
from matplotlib.tri import Triangulation
from examples.fracture2d.damage_norm_coupled_audit import peak_profile


class CrackProfileTest(unittest.TestCase):
    def test_straight_bent_and_no_crack_paths(self):
        x=np.linspace(.52,.98,47);y=np.linspace(.4,.6,81)
        X,Y=np.meshgrid(x,y,indexing='ij');tri=Triangulation(X.ravel(),Y.ravel())
        cuts=dict(x_grid=x,y_grid=np.linspace(.4,.6,401),threshold=.5)
        straight=np.exp(-((Y-.5)/.012)**2)
        a=peak_profile(straight.ravel(),tri,**cuts)
        self.assertEqual(a['active_cuts'],47)
        self.assertLess(a['max_abs_lateral_drift'],1e-12)
        np.testing.assert_allclose(a['half_height_width'],2*.012*np.sqrt(np.log(2)),atol=.001)
        center=.5+.02*np.sin(2*np.pi*(X-.52)/.46)
        b=peak_profile(np.exp(-((Y-center)/.012)**2).ravel(),tri,**cuts)
        self.assertGreater(b['max_abs_lateral_drift'],.018)
        self.assertGreater(b['rms_lateral_drift'],.01)
        absent=peak_profile((.01*straight).ravel(),tri,**cuts)
        self.assertEqual(absent['active_cuts'],0)
        self.assertIsNone(absent['max_abs_lateral_drift'])


if __name__=='__main__':unittest.main()
