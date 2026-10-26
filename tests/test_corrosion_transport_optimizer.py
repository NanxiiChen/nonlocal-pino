"""Schedule continuity and actual SOAP update checks for late refinement."""
import unittest
import jax.numpy as jnp
import numpy as np
from examples.corrosion2d.configs.residual_pilot import get_config
from examples.corrosion2d.transport_optimizer import make_tail_optimizer, make_tail_schedule
from nonlocal_pinn.training.optimizers import make_optimizer

class TailOptimizerTests(unittest.TestCase):
    def config(self):
        c=get_config('corrosion2d','phase',42)
        c.experiment.lr_tail_start=3000;c.experiment.lr_tail_end=1e-5
        return c

    def test_schedule_preserves_early_rates_and_reaches_explicit_end(self):
        c=self.config();tail=make_tail_schedule(c)
        c.experiment.lr_tail_start=0;base=make_tail_schedule(c)
        for step in (0,1,1000,2999,3000):
            np.testing.assert_array_equal(tail(step),base(step))
        self.assertAlmostEqual(float(tail(5000)),1e-5,places=10)
        self.assertGreater(float(tail(4999)),1e-5)
        self.assertLess(float(tail(4999)),1.01e-5)
        rates=np.array([tail(i) for i in range(0,5100,100)])
        self.assertTrue(np.all(np.diff(rates)<=0))
        for start,end in ((True,1e-5),(-1,1e-5),(5000,1e-5),(3000,0.),(3000,np.nan),(3000,1e-3)):
            c=self.config()
            with c.ignore_type():
                c.experiment.lr_tail_start=start;c.experiment.lr_tail_end=end
            with self.assertRaises(ValueError):make_tail_schedule(c)

    def test_actual_soap_updates_match_before_tail_and_shrink_after(self):
        c=self.config();c.train.iterations=5;c.experiment.lr_tail_start=3
        a,b=make_optimizer(c),make_tail_optimizer(c)
        params={'w':jnp.array([[.3,-.2],[.1,.5]])};grad={'w':jnp.array([[.2,.4],[-.5,.3]])}
        sa,sb=a.init(params),b.init(params)
        for step in range(5):
            ua,sa=a.update(grad,sa,params);ub,sb=b.update(grad,sb,params)
            if step<=3:np.testing.assert_array_equal(ua['w'],ub['w'])
            else:
                ratio=float(make_tail_schedule(c)(step))/float(make_tail_schedule(self.disabled(c))(step))
                np.testing.assert_allclose(ub['w'],ua['w']*ratio,rtol=2e-6,atol=1e-10)
                self.assertLess(np.linalg.norm(ub['w']),np.linalg.norm(ua['w']))

    def disabled(self,c):
        result=c.copy_and_resolve_references();result.experiment.lr_tail_start=0
        return result

if __name__=='__main__':unittest.main()
