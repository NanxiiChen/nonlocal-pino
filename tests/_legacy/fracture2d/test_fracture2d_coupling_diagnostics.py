"""Independent checks of reporting algebra and adjacent-sweep normalization."""

import unittest

import numpy as np

from examples.fracture2d._legacy.coupling_diagnostics import metrics


class CouplingDiagnosticsTest(unittest.TestCase):
    def test_manufactured_at2_decomposition_and_adjacent_changes(self):
        d = np.array([0.2, 0.4])
        H = np.array([2.0, 3.0])
        H_used = np.array([1.0, 2.0])
        local = 5.0 * d
        nonlocal_term = np.array([-0.2, 0.2])
        history = 2.0 * H * (d - 1.0)
        rho = local + nonlocal_term + history
        fields = {
            "q": np.array([0.0, 0.1, 0.2, 0.3]), "d": d, "H": H,
            "H_used": H_used, "at2_local": local,
            "at2_nonlocal": nonlocal_term, "at2_history": history,
            "at2_fixed_relaxed": local + nonlocal_term + 2.0 * H_used * (d - 1.0),
            "at2_fixed_solved": np.zeros(2),
            "coordinates": np.array([[0.0, 0.0], [1.0, 0.0]]),
        }
        report = {"sweep": 2, "load": 0.1, "reaction": 10.0,
                  "damage_inf": float(np.max(np.abs(rho))),
                  "mechanics_loss": 0.001, "gates": {"finite": True}}
        old_fields = {"q": fields["q"] - 0.01, "d": d - 0.02, "H": H - 0.1}
        result = metrics(report, fields, (old_fields, {"sweep": 1, "load": 0.1, "reaction": 9.0}))
        self.assertAlmostEqual(result["at2_relative_inf"], 2.4 / 5.8)
        self.assertLess(result["at2_decomposition_defect_inf"], 1e-14)
        self.assertEqual(result["at2_max_node"], 0)
        changes = result["adjacent_sweep"]
        self.assertAlmostEqual(changes["damage_change_inf"], 0.02)
        self.assertAlmostEqual(changes["q_change_relative_inf"], 0.1)
        self.assertAlmostEqual(changes["H_change_relative_inf"], 0.1 / 3.0)
        self.assertAlmostEqual(changes["reaction_change_relative"], 0.1)
        self.assertIsNone(metrics(report, fields, None)["adjacent_sweep"])
        fields["at2_local"] = local + 1.0
        with self.assertRaisesRegex(ValueError, "same-pass report"):
            metrics(report, fields, None)


if __name__ == "__main__":
    unittest.main()
