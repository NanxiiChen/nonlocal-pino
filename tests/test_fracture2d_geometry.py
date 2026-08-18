"""Pure-Python geometry checks for the fracture2d rounded slot."""

import unittest

import numpy as np

from examples.fracture2d.geometry import make_rounded_slot, points_inside_slot
from examples.fracture2d.configs.solver_base import get_config


class RoundedSlotGeometryTest(unittest.TestCase):
    def setUp(self):
        self.slot = make_rounded_slot(0.5, 0.5, 0.048, 0.024)

    def test_tip_and_center_follow_endpoint_convention(self):
        np.testing.assert_allclose(self.slot.tip, [0.5, 0.5])
        np.testing.assert_allclose(self.slot.center, [0.476, 0.5])
        self.assertAlmostEqual(self.slot.width, 2.0 * self.slot.radius)

    def test_points_are_classified_in_slot_and_material(self):
        points = np.asarray(
            [
                [0.1, 0.5],
                [0.475, 0.52],
                [0.499, 0.5],
                [0.52, 0.5],
                [0.476, 0.53],
            ]
        )
        inside = points_inside_slot(points, self.slot)
        np.testing.assert_array_equal(inside, [True, True, True, False, False])

    def test_rectangle_cap_join_remains_inside_slot(self):
        inside = points_inside_slot(
            np.asarray([[self.slot.center_x, self.slot.center_y]]),
            self.slot,
        )
        np.testing.assert_array_equal(inside, [True])

    def test_invalid_width_radius_pair_fails(self):
        with self.assertRaises(ValueError):
            make_rounded_slot(0.5, 0.5, 0.048, 0.02)

    def test_history_gate_is_explicitly_relaxed_for_2d_scale(self):
        config = get_config()
        self.assertEqual(config.boundary.mode, "bottom_vertical_point_horizontal")
        self.assertEqual(config.boundary.anchor_x, 0.0)
        self.assertEqual(config.boundary.anchor_y, 0.0)
        self.assertEqual(config.solver.gate_mode, "warning")
        self.assertEqual(config.solver.atol_history, 1.0e-5)
        self.assertEqual(config.solver.atol_d, 1.0e-5)
        self.assertEqual(config.solver.rtol_history, 1.0e-5)
        self.assertEqual(config.solver.atol_reaction, 1.0e-4)
        self.assertEqual(config.solver.rtol_reaction, 1.0e-6)
        np.testing.assert_allclose(
            config.output.field_grid_loads,
            [0.0055, 0.0058, 0.0059, 0.0060, 0.0061, 0.0062],
        )
