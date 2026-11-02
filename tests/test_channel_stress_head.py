"""Independent coupled stress-block least-squares quadratic and gradient."""
import unittest
import numpy as np
from docs.tasks.channel_stress_head_equilibration import statistics


class StressHead(unittest.TestCase):
    def test_coupled_quadratic_matches_residuals_and_finite_differences(self):
        rng=np.random.default_rng(37);x,xx,xy=rng.normal(size=(3,31,7))
        r,t=rng.normal(size=(2,31,2));wm,wc=rng.uniform(.01,1.,size=(2,31))
        G,g,f=statistics(x,xx,xy,r,t,wm,wc);v=rng.normal(size=14)
        def direct(a):
            c1,c2=a.reshape(2,7)
            residual=r+np.stack((-xx@c1-xy@c2,xy@c1-xx@c2),axis=-1)
            const=np.stack((x@c1,x@c2),axis=-1)-t
            return np.sum(wm[:,None]*residual**2)+np.sum(wc[:,None]*const**2)
        np.testing.assert_allclose(f+2*g@v+v@G@v,direct(v),atol=2e-12,rtol=2e-14)
        h=1e-5;derivative=np.array([(direct(v+np.eye(14)[i]*h)-direct(v-np.eye(14)[i]*h))/(2*h) for i in range(14)])
        np.testing.assert_allclose(2*(g+G@v),derivative,atol=2e-7,rtol=2e-8)
        self.assertGreater(np.linalg.eigvalsh(G)[0],0.)


if __name__=='__main__':unittest.main()
