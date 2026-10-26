"""Time curriculum reaches the full problem; disabled sampling replays exactly."""
import unittest

import jax
import numpy as np

from examples.corrosion2d.configs.residual_pilot import get_config
from examples.corrosion2d.residual_experiment import (
    sample_training_times, time_window_max, validate_time_window,
)


class TimeWindowTests(unittest.TestCase):
    def test_disabled_sampler_is_exact_legacy(self):
        c = get_config("electropolish2d", "phase", 42)
        validate_time_window(c)
        key = jax.random.PRNGKey(9)
        for step in (0, 25, 2999, 4999):
            old = jax.random.uniform(jax.random.fold_in(key, step), (c.sample.Nt,),
                                     minval=c.sample.t_min, maxval=1.0)
            np.testing.assert_array_equal(old, sample_training_times(c, key, step))

    def test_enabled_window_and_full_horizon_samples(self):
        c = get_config("electropolish2d", "latent", 42)
        c.loss.t_ramp_steps, c.loss.t_ramp_early = 3000, .05
        validate_time_window(c)
        self.assertEqual(time_window_max(c, 0), .05)
        self.assertAlmostEqual(time_window_max(c, 1500), .525)
        for step in (3000, 4999):
            self.assertEqual(time_window_max(c, step), 1.)
        key = jax.random.PRNGKey(8)
        for step in (0, 1000, 3000, 4999):
            times = np.asarray(sample_training_times(c, key, step))
            self.assertEqual(times.shape, (c.sample.Nt,))
            self.assertEqual(times.dtype, np.dtype("float32"))
            self.assertTrue(np.all(times >= c.sample.t_min))
            self.assertTrue(np.all(times < time_window_max(c, step)))
        c.loss.t_ramp_steps = 0
        np.testing.assert_array_equal(sample_training_times(c, key, 4000),
            jax.random.uniform(jax.random.fold_in(key, 4000), (c.sample.Nt,), minval=c.sample.t_min, maxval=1.))

    def test_invalid_window_is_rejected(self):
        for steps, early, late in ((True, .05, 1.), (-1, .05, 1.), (5000, .05, 1.),
                                    (3000, 0., 1.), (3000, .001, 1.), (3000, np.nan, 1.),
                                    (3000, 1., 1.), (3000, .05, .5)):
            c = get_config("electropolish2d", "phase", 42)
            with c.ignore_type():
                c.loss.t_ramp_steps, c.loss.t_ramp_early, c.loss.t_ramp_late = steps, early, late
            with self.assertRaises(ValueError):
                validate_time_window(c)


if __name__ == "__main__":
    unittest.main()
