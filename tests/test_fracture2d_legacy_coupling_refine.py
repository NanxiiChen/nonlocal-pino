"""New numerical controls must retain legacy updates and distinguish readiness."""

import unittest
import numpy as np
from ml_collections import ConfigDict

from examples.fracture2d.configs.train_base import get_config
from examples.fracture2d.stagger import _inner_update_ratio, _validate_stagger
from examples.fracture2d.legacy_coupling_refine import readiness, stage_diagnostic


class LegacyCouplingRefineTest(unittest.TestCase):
    def test_legacy_update_unchanged_and_tail_is_block_local(self):
        c = get_config()
        self.assertEqual(_inner_update_ratio(c, 1, 100), 1.)
        self.assertEqual(_inner_update_ratio(c, 100, 100), 1.)
        c.train.stagger_lr_policy = "linear_tail"
        c.train.stagger_tail_fraction = 0.25
        c.train.stagger_tail_ratio = 0.1
        _validate_stagger(c)
        self.assertEqual(_inner_update_ratio(c, 75, 100), 1.)
        self.assertAlmostEqual(_inner_update_ratio(c, 100, 100), 0.1)
        self.assertEqual(_inner_update_ratio(c, 1, 100), 1.)
        c.train.stagger_tail_ratio = 0.
        with self.assertRaisesRegex(ValueError, "stagger_tail_ratio"):
            _validate_stagger(c)

    def test_explicit_numerical_mixing_preserves_adopted_default(self):
        c = get_config()
        original = c.stagger.omega
        c.stagger.coupling_omega = 1.
        _validate_stagger(c)
        self.assertEqual(c.stagger.omega, original)
        c.stagger.coupling_omega = 0.
        with self.assertRaisesRegex(ValueError, "coupling omega"):
            _validate_stagger(c)

    def test_old_gate_passage_does_not_imply_coupled_readiness(self):
        limits = ConfigDict(dict(at2_relative_inf=1e-4, damage_change_inf=1e-4,
                                q_change_relative_inf=1e-4, H_change_relative_inf=1e-3,
                                reaction_change_relative=1e-3))
        diagnostic = {"at2_relative_inf": 0.0013537,
                      "adjacent_sweep": dict(damage_change_inf=0.0004165,
                          q_change_relative_inf=0.0005515, H_change_relative_inf=0.0024914,
                          reaction_change_relative=0.011138)}
        checks = readiness(diagnostic, {"finite": True, "mechanics": True}, limits)
        self.assertTrue(checks["old_physical_and_mechanics"])
        self.assertFalse(checks["coupled_at2"])
        self.assertFalse(checks["reaction_change_relative"])
        self.assertFalse(all(checks.values()))
        no_previous = readiness({"at2_relative_inf": 0., "adjacent_sweep": None}, {"finite": True}, limits)
        self.assertFalse(all(no_previous.values()))

    def test_stage_stop_checks_actual_adjacent_q_and_reaction(self):
        old = {"q": np.asarray([0., 0.001]), "d": np.asarray([0.1, 0.2]), "H": np.asarray([0.2, 0.3])}
        fields = {**{key: value.copy() for key, value in old.items()},
                  "at2_local": np.ones(2), "at2_nonlocal": np.zeros(2),
                  "at2_history": -np.ones(2), "reaction": 639.005}
        report = {"load": 0.006, "reaction": 639.}
        limits = ConfigDict(dict(at2_relative_inf=1e-4, damage_change_inf=1e-4,
                                q_change_relative_inf=1e-4, H_change_relative_inf=1e-3,
                                reaction_change_relative=1e-3))
        self.assertTrue(all(readiness(stage_diagnostic(fields, old, report), {"core_checked": True}, limits).values()))
        fields["q"][1] += 2e-6
        checks = readiness(stage_diagnostic(fields, old, report), {"core_checked": True}, limits)
        self.assertFalse(checks["q_change_relative_inf"])
        self.assertTrue(checks["coupled_at2"])


if __name__ == "__main__":
    unittest.main()
