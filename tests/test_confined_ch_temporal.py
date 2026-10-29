"""Time measure/curriculum invariants and actual L-BFGS node plumbing."""
from pathlib import Path
import tempfile,unittest
import jax,jax.numpy as jnp,equinox as eqx
import numpy as np
from examples.confined_ch2d.configs.train_coarsening import get_config as legacy
from examples.confined_ch2d.configs.train_coarsening_causal import get_config
from examples.confined_ch2d.problem import validate_config
from examples.confined_ch2d.sampling import temporal_samples
from examples.twophase2d.refinement import refine


class Amplitude(eqx.Module):
    a:object


class TemporalTests(unittest.TestCase):
    def test_legacy_rng_and_early_time_cdf_and_horizon(self):
        with jax.enable_x64(True):
            c=legacy();key=jax.random.PRNGKey(7)
            actual,H=temporal_samples(c,100,key)
            draws=jax.random.uniform(key,(c.sample.Nt,),dtype=jnp.float64,minval=1e-6,maxval=1-1e-6)
            expected=(jnp.arange(c.sample.Nt)+draws)*c.physical.terminal_time/c.sample.Nt
            np.testing.assert_array_equal(actual,expected);self.assertEqual(H,c.physical.terminal_time)
            c=get_config();validate_config(c)
            for iteration,H0 in [(1,.03),(3000,1.5),(4000,1.5)]:
                times,H=temporal_samples(c,iteration,key)
                self.assertAlmostEqual(H,H0)
                quantile=np.log1p(np.asarray(times)/c.sample.temporal.time_scale)/np.log1p(H/c.sample.temporal.time_scale)
                np.testing.assert_allclose(quantile,(np.arange(c.sample.Nt)+np.asarray(draws))/c.sample.Nt,atol=2e-15)
                self.assertTrue(np.all(np.diff(times)>0));self.assertGreater(float(times[0]),0.)
                self.assertLess(float(times[-1]),H)
            times,_=temporal_samples(c,1,key);self.assertLess(float(times[0]),.002)
            c.sample.temporal.warmup_iterations=0
            with self.assertRaises(ValueError):validate_config(c)

    def test_explicit_refinement_measure_changes_actual_optimum(self):
        c=get_config();c.train.refinement.iterations=20
        nodes=np.asarray(c.train.refinement.time_nodes)
        expected=float(np.sum(nodes**3)/np.sum(nodes**2))
        # y=a*t^2/2, y_t=t^2: one-parameter residual a*t-t^2.
        def loss(net,times,mean):return jnp.mean((net.a*times-times**2)**2),jnp.zeros(2)
        with jax.enable_x64(True),tempfile.TemporaryDirectory() as directory:
            model,report=refine(Amplitude(jnp.asarray(.2)),loss,jnp.asarray(0.),c,Path(directory),lambda model,i:None)
            np.testing.assert_allclose(model.a,expected,atol=1e-10)
            np.testing.assert_array_equal(report['time_points'],nodes)
        c.train.refinement.time_nodes=tuple(nodes[::-1])
        with self.assertRaises(ValueError):validate_config(c)


if __name__=='__main__':unittest.main()
