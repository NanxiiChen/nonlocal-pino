"""Geometry-operator time measures preserve the old draws and exact IC node."""
import unittest
import jax
import jax.numpy as jnp
import numpy as np
from examples.confined_ch2d.configs.train_operator_smallscale10_512_fourier import get_config as legacy
from examples.confined_ch2d.configs.train_operator_smallscale10_512_curriculum import get_config
from examples.confined_ch2d.operator_problem import validate
from examples.confined_ch2d.operator_training import operator_temporal_samples
from examples.confined_ch2d.sampling import temporal_samples


class OperatorTemporalTests(unittest.TestCase):
    def test_historical_draws_are_bitwise_unchanged(self):
        with jax.enable_x64(True):
            c=legacy();validate(c)
            for iteration in (1,3000,20000):
                key=jax.random.fold_in(jax.random.key(42),iteration)
                draws=jax.random.uniform(key,(c.sample.Nt,),dtype=jnp.float64)
                expected=jnp.concatenate((jnp.zeros(1),c.physical.terminal_time*(jnp.arange(c.sample.Nt)+draws)/c.sample.Nt))
                actual,H=operator_temporal_samples(c,iteration,key)
                np.testing.assert_array_equal(actual,expected)
                self.assertEqual(H,c.physical.terminal_time)

    def test_expanding_measure_has_fixed_shape_and_full_final_horizon(self):
        with jax.enable_x64(True):
            c=get_config();validate(c);key=jax.random.key(42)
            for iteration,expected_H in ((1,.15),(8000,3.),(20000,3.)):
                actual,H=operator_temporal_samples(c,iteration,key)
                positive,shared_H=temporal_samples(c,iteration,key)
                self.assertEqual(actual.shape,(c.sample.Nt+1,))
                self.assertEqual(float(actual[0]),0.)
                self.assertAlmostEqual(H,expected_H);self.assertEqual(H,shared_H)
                np.testing.assert_array_equal(actual[1:],positive)
                self.assertTrue(np.all(np.diff(actual)>0));self.assertLess(float(actual[-1]),H)
            for field,value in (('kind','unsupported'),('time_scale',0.),('initial_horizon',3.1),
                                ('warmup_iterations',1),('curriculum',1)):
                broken=get_config()
                with self.subTest(field=field),self.assertRaises((ValueError,TypeError)):
                    broken.sample.temporal[field]=value
                    validate(broken)


if __name__=='__main__':unittest.main()
