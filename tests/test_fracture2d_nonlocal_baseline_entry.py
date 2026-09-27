"""Focused tests for the single-entry direct nonlocal solver helpers."""

import unittest

import numpy as np

from examples.fracture2d import nonlocal_baseline as solver


class DirectNonlocalSolverHelperTest(unittest.TestCase):
    def test_adopted_scheme_constants_match_the_research_record(self) -> None:
        self.assertEqual(solver.SCHEME["omega"], 0.5)
        self.assertEqual(solver.SCHEME["max_stagger_iterations"], 144)
        self.assertEqual(solver.SCHEME["du_initial"], 1.0e-4)
        self.assertEqual(solver.SCHEME["du_min"], 6.25e-6)
        self.assertEqual(solver.SCHEME["acceptance_mode"], "stabilized_warning")
        self.assertEqual(solver.SCHEME["load_final"], 0.0075)
        self.assertEqual(solver.SCHEME["delta"], 0.048)

    def test_drift_mask_matches_amplification_envelope(self) -> None:
        loads = np.asarray([0.0, 1.0, 2.0])
        q = np.zeros((3, 3, 2), dtype=np.float64)
        q[1, 0] = (0.5, 0.0)
        q[1, 1] = (1.5, 0.0)
        q[2, 2] = (2.0, 0.0)
        retained = solver._drift_mask(loads, q, 1.08)
        self.assertTrue(retained[1, 0])
        self.assertFalse(retained[1, 1])
        self.assertTrue(retained[2, 2])
        self.assertTrue(retained[0].all())
        with self.assertRaisesRegex(ValueError, "material q"):
            solver._drift_mask(loads, q[..., 0], 1.08)

    def test_retained_range_excludes_drift_values(self) -> None:
        values = np.asarray([1.0, 5.0, 3.0])
        self.assertEqual(
            solver._retained_range(values, np.asarray([True, False, True])),
            (1.0, 3.0),
        )
        with self.assertRaisesRegex(ValueError, "removed every point"):
            solver._retained_range(values, np.zeros(3, dtype=bool))

    def test_warning_status_classification_is_explicit(self) -> None:
        self.assertTrue(solver._status_is_warning("initial_warning_accepted"))
        self.assertTrue(solver._status_is_warning("stabilized_warning_accepted"))
        self.assertFalse(solver._status_is_warning("initial"))
        self.assertFalse(solver._status_is_warning("stabilized_converged"))


if __name__ == "__main__":
    unittest.main()
