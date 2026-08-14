"""Regression tests for the direct fixed-history SparseD baseline."""

import unittest

import numpy as np

from examples.fracture1d.nonlocal_baseline import (
    _cell_history_from_displacement,
    _mass_lumped_cell_history_to_nodes,
    build_fixed_history_system,
    solve_fixed_history_damage,
    solve_fixed_history_defect,
    solve_pseudo_time_damage,
)
from pinn.operators import SparseD


class NonlocalBaselineTest(unittest.TestCase):
    def test_dg0_history_mass_lumping_is_the_nodal_average(self):
        cell_history = np.asarray([1.0, 3.0, 5.0])
        np.testing.assert_allclose(
            _mass_lumped_cell_history_to_nodes(cell_history),
            [1.0, 2.0, 4.0, 5.0],
        )

    def test_cell_history_uses_element_strain(self):
        x = np.linspace(0.0, 1.0, 4)
        displacement = x**2
        modulus = np.ones(3)
        previous = np.zeros(3)
        history = _cell_history_from_displacement(
            displacement, x, modulus, previous
        )
        expected_strain = np.diff(displacement) / np.diff(x)
        np.testing.assert_allclose(history, 0.5 * expected_strain**2)

    def test_fixed_history_solution_matches_dense_linear_system(self):
        x = np.linspace(0.0, 1.0, 41, dtype=np.float32)[:, None]
        operator = SparseD(x, 0.1, kernel="cubic")
        N_l = 0.02
        N_G = 1.0e-5
        history = np.full(len(x), 0.02, dtype=np.float64)
        damage, residual_mae, residual_inf = solve_fixed_history_damage(
            operator, N_l, N_G, history
        )
        D_dense = operator.to_scipy().toarray()
        factor = 2.0 * N_l / N_G
        system = (
            np.eye(len(x))
            + N_l**2 * D_dense
            + np.diag(factor * history)
        )
        expected = np.linalg.solve(system, factor * history)
        np.testing.assert_allclose(damage, expected, rtol=2e-6, atol=2e-6)
        self.assertLess(residual_mae, 2e-8)
        self.assertLess(residual_inf, 2e-7)

    def test_fixed_history_solution_has_no_hidden_postprocessing(self):
        x = np.linspace(0.0, 1.0, 41, dtype=np.float32)[:, None]
        operator = SparseD(x, 0.1, kernel="cubic")
        history = np.linspace(0.0, 0.03, len(x), dtype=np.float64)
        damage, _, _ = solve_fixed_history_damage(operator, 0.02, 1.0e-5, history)
        residual = damage + 0.02**2 * (operator.to_scipy() @ damage)
        residual -= (2.0 * 0.02 / 1.0e-5) * (1.0 - damage) * history
        np.testing.assert_allclose(residual, 0.0, atol=2e-7)

    def test_fixed_history_defect_has_correct_sign_and_correction(self):
        x = np.linspace(0.0, 1.0, 41, dtype=np.float32)[:, None]
        operator = SparseD(x, 0.1, kernel="cubic")
        scipy_operator = operator.to_scipy()
        N_l = 0.02
        N_G = 1.0e-5
        history = np.linspace(0.0, 0.03, len(x), dtype=np.float64)
        direct, _, _ = solve_fixed_history_damage(operator, N_l, N_G, history)
        perturbed = direct + 1.0e-3 * np.sin(np.linspace(0.0, 3.0, len(x)))
        system, rhs, _ = build_fixed_history_system(
            scipy_operator, N_l, N_G, history
        )
        defect, residual, solve_residual = solve_fixed_history_defect(
            scipy_operator, N_l, N_G, perturbed, history
        )
        np.testing.assert_allclose(system @ defect, residual, rtol=2e-6, atol=2e-7)
        np.testing.assert_allclose(solve_residual, 0.0, atol=2e-7)
        np.testing.assert_allclose(perturbed - defect, direct, rtol=2e-6, atol=2e-6)
        np.testing.assert_allclose(system @ direct, rhs, atol=2e-7)

    def test_pseudo_time_step_matches_implicit_dense_system(self):
        x = np.linspace(0.0, 1.0, 41, dtype=np.float32)[:, None]
        operator = SparseD(x, 0.1, kernel="cubic")
        N_l = 0.02
        N_G = 1.0e-5
        tau = 2.5e-4
        history = np.linspace(0.0, 0.03, len(x), dtype=np.float64)
        previous = np.linspace(0.1, 0.4, len(x), dtype=np.float64)
        scipy_operator = operator.to_scipy()
        damage, residual_mae, residual_inf = solve_pseudo_time_damage(
            operator,
            N_l,
            N_G,
            history,
            previous,
            tau,
            scipy_operator,
        )
        factor = 2.0 * N_l / N_G
        fixed_system = (
            np.eye(len(x))
            + N_l**2 * scipy_operator.toarray()
            + np.diag(factor * history)
        )
        expected = np.linalg.solve(
            np.eye(len(x)) + tau * fixed_system,
            previous + tau * factor * history,
        )
        np.testing.assert_allclose(damage, expected, rtol=2e-6, atol=2e-6)
        pseudo_residual = (
            (damage - previous) / tau
            + fixed_system @ damage
            - factor * history
        )
        np.testing.assert_allclose(pseudo_residual, 0.0, atol=2e-7)
        self.assertTrue(np.isfinite(residual_mae))
        self.assertTrue(np.isfinite(residual_inf))


if __name__ == "__main__":
    unittest.main()
