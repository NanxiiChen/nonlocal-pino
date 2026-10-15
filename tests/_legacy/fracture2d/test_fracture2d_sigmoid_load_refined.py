"""Load identity and committed-history invariants for critical-region refinement."""
import unittest
import jax
import jax.numpy as jnp
import numpy as np
from examples.fracture2d._legacy.sigmoid_damage_load_refined import committed_history, load_schedule


class SigmoidLoadRefinedTest(unittest.TestCase):
    def test_actual_loads_do_not_use_coarse_state_indices(self):
        values = load_schedule([209,210,211,212],2.5e-5,initial_load=.0052,end_load=.0053)
        np.testing.assert_allclose(values,[.005225,.005250,.005275,.005300],rtol=1e-14)
        self.assertNotAlmostEqual(values[0],53*1e-4,places=7)
        for ids in [[209,211,212],[208,209,210,211],[209.,210.,211.,212.]]:
            with self.assertRaises(ValueError):
                load_schedule(ids,2.5e-5,initial_load=.0052,end_load=.0053)

    def test_history_uses_previous_load_and_detaches_both_arguments(self):
        jax.config.update('jax_enable_x64',True)
        previous = jnp.array([2.,5.])
        first = committed_history(previous,jnp.array([8.,3.]))
        second = committed_history(previous,jnp.array([4.,1.]))
        np.testing.assert_array_equal(first,[8.,5.])
        np.testing.assert_array_equal(second,[4.,5.])
        # Within-load overshoot8 is not irreversibly committed; at next load,
        # the completed preceding-load value4 becomes the committed floor.
        np.testing.assert_array_equal(committed_history(second,jnp.array([3.,6.])),[4.,6.])
        gradients = jax.grad(lambda hp,driver:committed_history(hp,driver).sum(),argnums=(0,1))(
            previous,jnp.array([4.,1.]))
        for gradient in gradients:
            np.testing.assert_array_equal(gradient,jnp.zeros_like(gradient))


if __name__ == '__main__':
    unittest.main()
