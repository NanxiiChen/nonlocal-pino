"""Tests for the fixed-history transition continuation helper."""

import unittest
from unittest import mock

import numpy as np

from examples.fracture1d.transition_diagnostics import (
    _bisect_history_interval,
    _build_objective_optimizers,
    _history_continuation_values,
)
from examples.fracture1d.configs.train_base import get_config


class TransitionDiagnosticTest(unittest.TestCase):
    def test_history_path_has_fixed_endpoints(self):
        previous = np.asarray([0.0, 1.0, 2.0], dtype=np.float64)
        target = np.asarray([1.0, 3.0, 6.0], dtype=np.float64)
        fractions, histories = _history_continuation_values(previous, target, 4)

        np.testing.assert_allclose(fractions, [0.25, 0.5, 0.75, 1.0])
        np.testing.assert_allclose(histories[0], previous + 0.25 * (target - previous))
        np.testing.assert_allclose(histories[-1], target)

    def test_history_path_rejects_invalid_inputs(self):
        previous = np.zeros(3, dtype=np.float64)
        target = np.ones(3, dtype=np.float64)
        with self.assertRaises(ValueError):
            _history_continuation_values(previous, target, 0)
        with self.assertRaises(ValueError):
            _history_continuation_values(previous, np.ones(2), 2)
        with self.assertRaises(ValueError):
            _history_continuation_values(previous.astype(np.int32), target, 2)

    def test_history_interval_bisection_is_strictly_inside(self):
        midpoint = _bisect_history_interval(0.25, 0.5)
        self.assertAlmostEqual(midpoint, 0.375)
        with self.assertRaises(ValueError):
            _bisect_history_interval(0.5, 0.5)
        with self.assertRaises(ValueError):
            _bisect_history_interval(-0.1, 0.5)

    def test_residual_optimizer_uses_explicit_learning_rate(self):
        config = get_config()
        built_learning_rates = []

        def record_optimizer(current_config):
            built_learning_rates.append(float(current_config.train.lr_init))
            return object()

        with mock.patch(
            "examples.fracture1d.transition_diagnostics.train._build_optimizer",
            side_effect=record_optimizer,
        ):
            energy_optimizer, residual_optimizer, energy_lr = (
                _build_objective_optimizers(config, 2.0e-4, "adam", 1.0e-5)
            )

        self.assertIsNot(energy_optimizer, residual_optimizer)
        self.assertEqual(built_learning_rates, [2.0e-3, 2.0e-4])
        self.assertEqual(energy_lr, 2.0e-3)
        self.assertEqual(float(config.train.lr_init), 2.0e-3)


if __name__ == "__main__":
    unittest.main()
