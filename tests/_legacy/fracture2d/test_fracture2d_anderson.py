"""Check the sourced Type-II formula and explicit safeguard/restart behavior."""

import unittest
import numpy as np

from examples.fracture2d._legacy.anderson import propose, safeguarded_step


class AndersonTest(unittest.TestCase):
    def test_proposal_matches_independent_least_squares(self):
        rng = np.random.default_rng(73)
        xs, fs = rng.normal(size=(3, 8)), rng.normal(size=(3, 8))
        beta = 0.5
        actual, info = propose(list(xs[:2]), list(fs[:2]), xs[2], xs[2] + fs[2], depth=5, beta=beta)
        dx, df = np.diff(xs, axis=0).T, np.diff(fs, axis=0).T
        gamma = np.linalg.lstsq(df, fs[2], rcond=None)[0]
        expected = xs[2] + beta * fs[2] - (dx + beta * df) @ gamma
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
        self.assertEqual(info["qr_rank"], 2)

    def test_acceleration_and_failure_propagation(self):
        diagonal = np.asarray([0.95, 0.9, 0.8])
        target = np.asarray([0.2, 0.3, 0.4])
        def evaluate(x):
            return diagonal * x + (1 - diagonal) * target, {"physical": True}
        x, hx, hf = np.zeros(3), [], []
        selected = []
        for _ in range(20):
            image, _ = evaluate(x)
            x, _, _, info, hx, hf = safeguarded_step(x, image, hx, hf,
                evaluate=evaluate, bounds_pass=lambda d: min(d) >= 0 and max(d) <= 1,
                physics_pass=lambda p: p["physical"], depth=3, beta=0.5,
                sufficient_decrease=0.01, max_backtracks=4, restart_policy="reject_or_rank")
            selected.append(info["selected_kind"])
        np.testing.assert_allclose(x, target, rtol=0, atol=1e-10)
        self.assertIn("anderson", selected)
        def fail(_):
            raise OSError("diagnostic I/O failure")
        with self.assertRaisesRegex(OSError, "I/O failure"):
            safeguarded_step(np.zeros(3), np.ones(3) * 0.1, [], [], evaluate=fail,
                bounds_pass=lambda _: True, physics_pass=lambda _: True,
                depth=3, beta=0.5, sufficient_decrease=0.01, max_backtracks=4, restart_policy="reject_or_rank")

    def test_rank_restart_and_physical_rejection(self):
        x, image = np.asarray([0.2, 0.3]), np.asarray([0.3, 0.4])
        candidate, info = propose([x.copy()], [image - x], x, image, depth=3, beta=0.5)
        self.assertIsNone(candidate)
        self.assertEqual(info["proposal_reason"], "rank_deficient")
        result = safeguarded_step(x, image, [x.copy()], [image - x],
            evaluate=lambda d: (d + 0.05, {"physical": False}),
            bounds_pass=lambda _: True, physics_pass=lambda p: p["physical"],
            depth=3, beta=0.5, sufficient_decrease=0.01, max_backtracks=4, restart_policy="reject_or_rank")
        self.assertTrue(result[3]["restart"])
        self.assertFalse(result[3]["selected_physics_pass"])
        np.testing.assert_array_equal(result[0], x + 0.5 * (image - x))

    def test_nonlinear_backtracking_preserves_unclipped_proposal(self):
        def evaluate(x):
            return np.minimum(0.11 + 0.9 * x, 0.38), {"x": float(x[0])}
        past, x = np.asarray([0.1]), np.asarray([0.2])
        image, _ = evaluate(x)
        past_image, _ = evaluate(past)
        for physical_limit in (False, True):
            with self.subTest(physical_limit=physical_limit):
                result = safeguarded_step(x, image, [past], [past_image - past],
                    evaluate=evaluate,
                    bounds_pass=(lambda d: True) if physical_limit else (lambda d: 0 <= d[0] <= 1),
                    physics_pass=(lambda p: p["x"] <= 0.5) if physical_limit else (lambda _: True),
                    depth=3, beta=0.5, sufficient_decrease=0.01, max_backtracks=4, restart_policy="reject_or_rank")
                self.assertEqual(result[3]["selected_kind"], "anderson")
                self.assertEqual(result[3]["trials"][0]["reason"], "physics" if physical_limit else "bounds")
                self.assertEqual(result[3]["trials"][-1]["alpha"], 0.25)
                self.assertAlmostEqual(result[0][0], 0.45875)

    def test_rank_only_retains_valid_evaluated_points_after_rejection(self):
        past, x = np.asarray([0.1]), np.asarray([0.2])
        image = np.asarray([0.28])
        common = dict(evaluate=lambda d: (np.minimum(0.28 + 30 * (d - 0.2), 0.9), {}),
                      bounds_pass=lambda _: True, physics_pass=lambda _: True,
                      depth=5, beta=0.5, sufficient_decrease=0.01, max_backtracks=4)
        old = safeguarded_step(x, image, [past], [np.asarray([0.1])], restart_policy="reject_or_rank", **common)
        retained = safeguarded_step(x, image, [past], [np.asarray([0.1])], restart_policy="rank_only", **common)
        self.assertEqual(old[3]["selected_kind"], "picard")
        self.assertEqual(retained[3]["selected_kind"], "picard")
        self.assertEqual(len(old[4]), 1)
        self.assertEqual(len(retained[4]), 2)
        np.testing.assert_array_equal(old[0], retained[0])

    def test_deficient_oldest_secant_is_dropped_without_losing_recent_direction(self):
        x = np.asarray([0.2, 0.3, 0.4])
        previous = np.zeros(3)
        previous_f = np.ones(3) * 0.1
        image = x + np.asarray([0.2, 0.3, 0.5])
        history_x, history_f = [previous, previous.copy()], [previous_f, previous_f.copy()]
        old, _ = propose(history_x, history_f, x, image, depth=5, beta=0.5)
        self.assertIsNone(old)
        candidate, info = propose(history_x, history_f, x, image, depth=5, beta=0.5, rank_strategy="drop_oldest")
        expected, _ = propose([previous], [previous_f], x, image, depth=5, beta=0.5)
        self.assertIsNotNone(candidate)
        np.testing.assert_allclose(candidate, expected, rtol=0, atol=0)
        self.assertEqual(info["discarded_oldest"], 1)
        self.assertEqual(info["history_size"], 1)


if __name__ == "__main__":
    unittest.main()
