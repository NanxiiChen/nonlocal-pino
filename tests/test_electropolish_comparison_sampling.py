"""Common prescribed fields must not depend on a derivative method's query set."""
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from examples.electropolish2d.comparison_sampling import (
    make_comparison_sampler, prescribed_shared_fields,
)
from examples.electropolish2d.configs.train_operator_threeway import get_config
from examples.electropolish2d.online_conditions import initial_fields, make_sampler


class ComparisonSamplingTests(unittest.TestCase):
    def test_common_inputs_match_exactly_with_different_core_halo_layouts(self):
        base = get_config()
        base.operator.condition_batch = 2
        base.operator.image_shape = (1, 4, 8)
        base.operator.ic_points = 7
        base.sample.Nx, base.sample.Ny = 7, 5
        rng = np.random.default_rng(11)
        initial = rng.uniform([0, .1], [1, .4], (7, 2)).astype(np.float32)
        xs = np.linspace(0, 1, 7, dtype=np.float32)
        boundary = np.r_[np.c_[xs, np.zeros(7)], np.c_[xs, np.full(7, .5)]].astype(np.float32)
        common = np.concatenate((initial, boundary))
        records = []
        for method, core_count in (('sparse', 19), ('local_ad', 7), ('fem_weak', 11)):
            c = base.copy_and_resolve_references()
            c.nl.method = method
            core = rng.uniform([0, 0], [1, .5], (core_count, 2)).astype(np.float32)
            side = np.array([[0, .1], [0, .25], [0, .4], [1, .1], [1, .25], [1, .4]], np.float32)
            points = jnp.asarray(np.concatenate((core, common, side if method == 'local_ad' else side[:0])))
            sample, geometry = make_comparison_sampler(c)
            key = jax.random.key(42)
            image, known, coefficients, count = sample(key, points, geometry(points))
            original, original_geometry = make_sampler(c)
            legacy = original(key, points, original_geometry(points))
            np.testing.assert_array_equal(coefficients, legacy[2])
            np.testing.assert_array_equal(count, legacy[3])
            np.testing.assert_array_equal(known[:, :core_count], legacy[1][:, :core_count])
            shared = np.asarray(known[:, core_count:core_count + len(common)])
            np.testing.assert_array_equal(shared[:, 7:14], 1.)
            np.testing.assert_array_equal(shared[:, 14:21], 0.)
            records.append((np.asarray(image), shared, np.asarray(coefficients), np.asarray(count)))
        for current in records[1:]:
            for a, b in zip(records[0], current, strict=True):
                np.testing.assert_array_equal(a, b)

    def test_host_values_agree_with_independent_double_precision_jax_form(self):
        c = get_config()
        coefficients = np.array([[[.001, 3.1, .2], [.002, 2.3, .5], [.001, 4.2, .3],
                                  [.001, 3.8, 2.], [.002, 4.1, 1.], [.001, 3.4, 4.]]], np.float32)
        points = np.array([[.17, .35], [.4, .355], [.72, .345], [.5, 0], [.5, .5]], np.float32)
        actual = prescribed_shared_fields(coefficients, points, c)
        with jax.enable_x64(True):
            curve = np.linspace(0, 1, 2000).astype(np.float32).astype(np.float64)
            k = np.sqrt(c.physical.omega_phi / (2 * c.physical.alpha_phi)) * c.nondim.L_c
            expected = initial_fields(jnp.asarray(coefficients[0], dtype=jnp.float64),
                jnp.asarray(points, dtype=jnp.float64), jnp.asarray(curve),
                jnp.asarray(np.searchsorted(curve, points[:, 0]), dtype=jnp.int32),
                y0=c.surface.y0, k=k, c_se=c.physical.c_se, window=16)
        np.testing.assert_allclose(actual[0], np.asarray(expected), rtol=6e-8, atol=1e-12)
        self.assertEqual(actual.dtype, np.float32)
        invalid = get_config()
        invalid.operator.online.input_evaluation = 'unknown'
        with self.assertRaisesRegex(ValueError, 'declared image-only comparison'):
            make_comparison_sampler(invalid)


if __name__ == '__main__':
    unittest.main()
