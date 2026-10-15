"""Diagnostic readiness is rechecked before committing a next-load history."""

import json
from pathlib import Path
import tempfile
import unittest

from ml_collections import ConfigDict
import numpy as np

from examples.fracture2d._legacy.legacy_coupled_loading import (
    LIMITS, _preserved_settings, coupling_callbacks, load_diagnostic, require_ready_parent, verify_confirmations,
)
from examples.fracture2d._legacy.configs.train_base import get_config


class CoupledParentTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.config = ConfigDict(dict(
            coupling={**LIMITS, "after_stop": True},
            stagger=dict(sweeps=4, damage_range_atol=1e-10, healing_atol=1e-3,
                mechanics_loss_limit=.01, reaction_relative_limit=.05,
                gate_displacement_l2=False, damage_rmse_limit=.01)))
        self.state = (None, None, None, None, np.array([0., .006]),
                      np.array([.1, .2]), np.array([.3, .4]))
        self.fields = dict(q=self.state[4], d=self.state[5], H=self.state[6],
            H_used=self.state[6], d_solved=self.state[5], d_previous=self.state[5],
            at2_local=np.ones(2), at2_nonlocal=np.zeros(2), at2_history=-np.ones(2),
            at2_fixed_relaxed=np.zeros(2), at2_fixed_solved=np.zeros(2),
            coordinates=np.array([[0., 0.], [1., 0.]]))
        self.report = dict(load=.006, reaction=639., mechanics_loss=1e-5,
            damage_inf=0., damage_min=.1, damage_max=.2, damage_healing_max=0.,
            history_decrease_max=0., drift_count=0, accepted=False,
            diagnostic_only=True, reaction_relative=0., d_rmse=0.,
            gates=dict(finite=True, mechanics_loss=True, damage_range=True,
                       healing=True, history_monotonic=True, drift=True))

    def write_sweeps(self):
        observe, _ = coupling_callbacks(self.root, self.config.coupling)
        for sweep in range(1, 5):
            report = {**self.report, "sweep": sweep}
            (self.root / f"sweep_{sweep:04d}.json").write_text(json.dumps(report))
            observe(report, self.fields)

    def test_recomputed_three_confirmations_and_checkpoint_pairing(self):
        self.write_sweeps()
        with self.assertRaisesRegex(ValueError, "networks/Adam and q/d/H"):
            verify_confirmations(self.root, self.config, self.state[:-1], require_ready=True)
        final = verify_confirmations(self.root, self.config, self.state, require_ready=True)
        self.assertEqual(final["consecutive"], 3)
        changed = (*self.state[:6], self.state[6] + .001)
        with self.assertRaisesRegex(ValueError, "checkpoint H"):
            verify_confirmations(self.root, self.config, changed, require_ready=True)

    def test_forged_ready_flag_or_diagnostic_is_rejected(self):
        self.write_sweeps()
        path = self.root / "readiness_0004.json"
        value = json.loads(path.read_text())
        value["consecutive"] = 4
        path.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, "readiness differs"):
            verify_confirmations(self.root, self.config, self.state, require_ready=True)

    def test_wrong_sha_is_rejected_before_loading_parent(self):
        (self.root / "diagnostic.ckpt.zip").write_bytes(b"changed diagnostic")
        with self.assertRaisesRegex(ValueError, "SHA256 differs"):
            load_diagnostic(self.root, "0" * 64, 60, None, None, None, None, require_ready=True)

    def test_modified_coupling_evidence_is_rejected(self):
        self.write_sweeps()
        path = self.root / "coupling_0004.json"
        value = json.loads(path.read_text())
        value["at2_relative_inf"] = 1e-8
        path.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, "diagnostics differ"):
            verify_confirmations(self.root, self.config, self.state, require_ready=True)

    def test_numerical_controls_cannot_change_kernel_budget_or_readiness(self):
        source = get_config()
        source.train.dtype = "float64"
        with self.assertRaisesRegex(ValueError, "explicitly stored"):
            _preserved_settings(source, source)
        config = ConfigDict(json.loads(source.to_json_best_effort()))
        config.coupling = ConfigDict({**LIMITS, "after_stop": True})
        config.train.stagger_lr_policy = "linear_tail"
        config.train.stagger_tail_fraction = .25
        config.train.stagger_tail_ratio = .01
        config.stagger.coupling_omega = 1.
        config.stagger.sweeps = 16
        _preserved_settings(config, source)
        config.nl.delta *= 2
        with self.assertRaisesRegex(ValueError, "preserved nl"):
            _preserved_settings(config, source)
        config.nl.delta = source.nl.delta
        config.stagger.mechanics_steps += 1
        with self.assertRaisesRegex(ValueError, "stage budgets"):
            _preserved_settings(config, source)
        config.stagger.mechanics_steps = source.stagger.mechanics_steps
        config.coupling.at2_relative_inf = 1e-3
        with self.assertRaisesRegex(ValueError, "limits differ"):
            _preserved_settings(config, source)

    def test_unready_trial_may_seed_same_load_but_not_advance(self):
        self.fields["at2_history"] = -np.ones(2) + .01
        self.fields["at2_fixed_relaxed"] = np.full(2, .01)
        self.fields["at2_fixed_solved"] = np.full(2, .01)
        self.report["damage_inf"] = .010000000000000009
        self.write_sweeps()
        self.assertFalse(verify_confirmations(self.root, self.config, self.state,
                                             require_ready=False)["coupled_ready"])
        with self.assertRaisesRegex(ValueError, "three consecutive"):
            verify_confirmations(self.root, self.config, self.state, require_ready=True)

    def test_callbacks_recover_counter_and_same_pass_stop_from_files(self):
        self.write_sweeps()
        observe, stop = coupling_callbacks(self.root, self.config.coupling)
        self.assertFalse(stop(1, self.fields))
        current = {**self.fields, "reaction": 639.}
        self.assertTrue(stop(5, current))
        report = {**self.report, "sweep": 5}
        (self.root / "sweep_0005.json").write_text(json.dumps(report))
        observe(report, self.fields)
        self.assertEqual(json.loads((self.root / "readiness_0005.json").read_text())["consecutive"], 4)
        current["q"] = self.fields["q"] + 1e-5
        self.assertFalse(stop(5, current))

    def test_warning_ancestry_is_explicit_and_does_not_change_limits(self):
        self.assertTrue(require_ready_parent(self.config))
        self.config.coupling.allow_unready_parent = True
        self.assertFalse(require_ready_parent(self.config))
        self.assertEqual(self.config.coupling.q_change_relative_inf, 1e-4)
        self.config.coupling.allow_unready_parent = False
        self.assertTrue(require_ready_parent(self.config))
        invalid = ConfigDict(dict(coupling=dict(allow_unready_parent="true")))
        with self.assertRaisesRegex(ValueError, "must be boolean"):
            require_ready_parent(invalid)

    def test_warning_seed_still_refuses_failed_physical_mechanics_gates(self):
        self.report["mechanics_loss"] = .02
        self.report["gates"]["mechanics_loss"] = False
        self.write_sweeps()
        with self.assertRaisesRegex(ValueError, "original physical/mechanics gates"):
            verify_confirmations(self.root, self.config, self.state, require_ready=False)


if __name__ == "__main__":
    unittest.main()
