"""Tests for fixed-history local hat-basis corrections."""

import unittest

import numpy as np

from examples.fracture1d.local_enrichment import (
    make_hat_basis,
    solve_local_damage_correction,
)
from examples.fracture1d.nonlocal_baseline import solve_fixed_history_damage
from pinn.operators import SparseD


class LocalEnrichmentTest(unittest.TestCase):
    def test_hat_basis_is_nodal_and_contiguous(self):
        x = np.linspace(0.0, 1.0, 9)
        basis, indices = make_hat_basis(x, 4, 3)
        np.testing.assert_array_equal(indices, [3, 4, 5])
        np.testing.assert_allclose(basis[indices, :], np.eye(3))
        self.assertEqual(np.count_nonzero(basis[:, 1]), 1)

    def test_local_correction_recovers_known_nodal_perturbation(self):
        x = np.linspace(0.0, 1.0, 41, dtype=np.float32)[:, None]
        operator = SparseD(x, 0.1, kernel="cubic").to_scipy()
        N_l = 0.02
        N_G = 1.0e-5
        history = np.full(len(x), 0.02, dtype=np.float64)
        target, _, _ = solve_fixed_history_damage(
            SparseD(x, 0.1, kernel="cubic"), N_l, N_G, history
        )
        basis, _ = make_hat_basis(x[:, 0], 20, 3)
        known_coefficients = np.asarray([0.02, -0.04, 0.01])
        base = target - basis @ known_coefficients
        corrected, coefficients, _, raw_inf, _, scaled_inf = (
            solve_local_damage_correction(
                operator,
                N_l,
                N_G,
                history,
                base,
                basis,
                "jacobi",
            )
        )
        np.testing.assert_allclose(corrected, target, rtol=2e-7, atol=2e-7)
        np.testing.assert_allclose(
            coefficients, known_coefficients, rtol=2e-6, atol=2e-6
        )
        self.assertLess(raw_inf, 1e-5)
        self.assertLess(scaled_inf, 1e-7)

    def test_hat_basis_requires_odd_patch(self):
        with self.assertRaises(ValueError):
            make_hat_basis(np.linspace(0.0, 1.0, 9), 4, 2)


if __name__ == "__main__":
    unittest.main()
