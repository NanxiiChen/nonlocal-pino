"""Cold loading grid and irreversible-history regressions."""
import jax
import jax.numpy as jnp
import numpy as np
import unittest

from examples.fracture2d._legacy.configs.l1_damage_loading_from_zero import get_config
from examples.fracture2d._legacy.l1_damage_loading import committed_history, loading_grid, validate


class ColdL1LoadingTest(unittest.TestCase):
    def test_requested_grid_uses_matching_loads_and_includes_transition_once(self):
        grid = validate(get_config())
        assert grid.shape == (20,)
        np.testing.assert_allclose(grid[:10], np.arange(1, 11)*5e-4, rtol=0, atol=1e-14)
        np.testing.assert_allclose(grid[10:], np.arange(51, 61)*1e-4, rtol=0, atol=1e-14)
        assert np.count_nonzero(np.isclose(grid, .0050, rtol=0, atol=1e-14)) == 1
        with self.assertRaisesRegex(ValueError, 'increment grids'):
            loading_grid(transition_load=.0051, coarse_step=5e-4, fine_step=1e-4, end_load=.006)


    def test_history_uses_parent_load_floor_and_is_detached(self):
        parent = jnp.array([1., 2., 3.])
        first = committed_history(parent, jnp.array([5., 1., 7.]))
        second = committed_history(parent, jnp.array([4., 0., 6.]))
        np.testing.assert_array_equal(first, [5., 2., 7.])
        np.testing.assert_array_equal(second, [4., 2., 6.])
        np.testing.assert_array_equal(jax.grad(lambda driver: committed_history(parent, driver).sum())(
            jnp.array([4., 0., 6.])), np.zeros(3))


    def test_no_silent_protocol_switch_or_zero_history_training(self):
        c = get_config()
        assert c.l1_loading.initial_physical_damage == c.l1_loading.initial_physical_history == 0
        assert c.l1_loading.minimum_rate == c.l1_loading.learning_rate == 1e-5
        c.l1_loading.explicit_residual_weighting = True
        with self.assertRaisesRegex(ValueError, 'declared'):
            validate(c)
        c = get_config()
        c.l1_loading.inner_updates = 513
        with self.assertRaisesRegex(ValueError, 'declared'):
            validate(c)


if __name__ == "__main__":
    unittest.main()
