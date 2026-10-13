"""Accuracy, stability, budget and phase-recovery contracts for outer gates."""

from argparse import Namespace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import jax
from ml_collections import ConfigDict
import numpy as np

from examples.fracture2d.outer_stopping import evaluate, validate_controls
from examples.fracture2d.configs.hybrid_pinn_stagger_gated import get_config
from examples.fracture2d.hybrid_pinn import SCHEMA_GATED, validate_completed


class OuterStoppingTest(unittest.TestCase):
    def controls(self):
        c = get_config()
        c.outer_stop.min_sweeps = 2
        c.outer_stop.max_sweeps = 4
        c.outer_stop.consecutive_passes = 2
        return c

    def report(self):
        return dict(fixed_H_image_AT2_relative=1e-4, coupled_AT2_relative=1e-4,
            damage_change=1e-6, q_change_over_load=1e-5, H_change_relative=1e-5,
            reaction_change_relative=1e-5, healing=0., drift_count=0, history_decrease=0.,
            damage_min=0., damage_max=.5, fixed_H_image_damage_min=0., fixed_H_image_damage_max=.5,
            mechanics=dict(converged=True, free_residual_norm=1e-9, convergence_limit=1e-8))

    def test_consecutive_accuracy_and_stability_not_plateau_only(self):
        c = self.controls()
        validate_controls(c.outer_stop)
        report = self.report()
        first = {**report, "q_change_over_load": None, "H_change_relative": None,
                 "reaction_change_relative": None}
        a = evaluate(first, c.outer_stop, c.reporting, sweep=1, previous_consecutive=0)
        self.assertEqual(a["consecutive"], 0)
        b = evaluate(report, c.outer_stop, c.reporting, sweep=2, previous_consecutive=0)
        self.assertFalse(b["converged"])
        ready = evaluate(report, c.outer_stop, c.reporting, sweep=3, previous_consecutive=1)
        self.assertTrue(ready["converged"])
        inaccurate = {**report, "fixed_H_image_AT2_relative": .1}
        failed = evaluate(inaccurate, c.outer_stop, c.reporting, sweep=4, previous_consecutive=1)
        self.assertFalse(failed["converged"])
        self.assertEqual(failed["consecutive"], 0)
        self.assertEqual(failed["reason"], "budget_exhausted")
        self.assertFalse(failed["checks"]["fixed_H_image_AT2"]["pass"])

    def test_physical_force_and_nonfinite_checks(self):
        c = self.controls()
        for key, value in (("drift_count", 1), ("healing", .1), ("history_decrease", .01),
                           ("fixed_H_image_damage_max", 1.01)):
            failed = evaluate({**self.report(), key: value}, c.outer_stop, c.reporting,
                              sweep=3, previous_consecutive=1)
            self.assertFalse(failed["converged"])
        with self.assertRaises(FloatingPointError):
            evaluate({**self.report(), "q_change_over_load": float("nan")}, c.outer_stop,
                     c.reporting, sweep=2, previous_consecutive=0)
        with self.assertRaises(FloatingPointError):
            evaluate({**self.report(), "fixed_H_image_damage_max": float("nan")}, c.outer_stop,
                     c.reporting, sweep=2, previous_consecutive=0)
        bad = self.report()
        bad["mechanics"]["free_residual_norm"] = -1.
        with self.assertRaisesRegex(ValueError, "mechanical diagnostics"):
            evaluate(bad, c.outer_stop, c.reporting, sweep=2, previous_consecutive=0)

    def test_controls_and_variable_completed_sweep_counter(self):
        c = self.controls()
        with self.assertRaisesRegex(ValueError, "boolean"):
            validate_controls(ConfigDict(dict(enabled="yes")))
        c.outer_stop.max_sweeps = 2
        with self.assertRaisesRegex(ValueError, "budget"):
            validate_controls(c.outer_stop)
        c = self.controls()
        ready = evaluate(self.report(), c.outer_stop, c.reporting, sweep=3, previous_consecutive=1)
        report = dict(outer_convergence=ready)
        metadata = dict(phase="complete", sweep=3, step=2)
        validate_completed(metadata, report, c.to_dict())
        with self.assertRaisesRegex(ValueError, "not stagger-ready"):
            validate_completed({**metadata, "step": 0}, report, c.to_dict())
        with self.assertRaisesRegex(ValueError, "not stagger-ready"):
            validate_completed(metadata, {"outer_convergence": {**ready, "converged": False}}, c.to_dict())

    @unittest.skipUnless(any(d.platform == "gpu" for d in jax.devices()), "GPU integration")
    def test_gated_phase_resume_fork_extension_and_unready_parent_exclusion(self):
        from examples.fracture2d import hybrid_pinn
        from examples.fracture2d.split_reference import run as run_reference
        from examples.fracture2d.split_pinn import neural_state_root
        from tests.test_fracture2d_split_horizon import square_cloud
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            root = Path(directory)
            c = get_config()
            cloud = root / "cloud.npz"
            np.savez(cloud, **square_cloud())
            c.data.cloud = str(cloud)
            c.data.cloud_sha256 = hashlib.sha256(cloud.read_bytes()).hexdigest()
            c.data.reference_root, c.data.reference_policy = str(root / "reference"), "strict"
            c.data.reference_indices = [1, 2, 3, 4]
            c.data.end_state_index = 2
            c.nl.delta_u, c.nl.delta_d = .4, .6
            c.geometry.grip_extent, c.geometry.h_grip = .6, .2
            c.arch.hidden_dim, c.arch.num_layers = 8, 1
            c.arch.use_spatial_emb = False
            c.stagger.damage_steps = 2
            c.output.log_every = 2
            c.output.fork_from, c.output.fork_after_index = "", 0
            c.output.dir, c.output.event_dir = str(root / "gated"), str(root / "events")
            c.outer_stop.min_sweeps, c.outer_stop.max_sweeps, c.outer_stop.consecutive_passes = 2, 6, 2
            c.outer_stop.inner_AT2_relative_limit = 2.
            c.outer_stop.q_change_limit = c.outer_stop.H_change_limit = c.outer_stop.reaction_change_limit = 1.
            c.reporting.coupled_AT2_relative_limit = 2.
            c.reporting.damage_change_limit = 1.
            config = root / "config.py"
            config.write_text("import json\nfrom ml_collections import ConfigDict\n"
                "def get_config():\n    return ConfigDict(json.loads(" + repr(c.to_json_best_effort()) + "))\n")
            run_reference(Namespace(config=config, output=root / "reference", load_step=.0001,
                last_load=.0004, max_sweeps=50, resume=False, fork_from=None,
                fork_after_index=0, omega=.5, trial_from=None))
            legacy = ConfigDict(c.to_dict())
            del legacy.outer_stop
            legacy.data.end_state_index = 1
            legacy.output.dir, legacy.output.event_dir = str(root / "legacy"), str(root / "legacy_events")
            hybrid_pinn.run(legacy)
            c.output.fork_from, c.output.fork_after_index = str(root / "legacy"), 1
            original_write = hybrid_pinn.atomic_json
            def interrupt_report(path, payload):
                if path.name == "sweep_0002.json" and path.parent.name == "state_0002":
                    raise OSError("manufactured gated report interruption")
                return original_write(path, payload)
            with patch.object(hybrid_pinn, "atomic_json", side_effect=interrupt_report):
                with self.assertRaisesRegex(OSError, "gated report interruption"):
                    hybrid_pinn.run(c)
            with zipfile.ZipFile(root / "gated/state_0002/progress.zip") as f:
                meta = json.loads(f.read("metadata.json"))
                self.assertEqual(meta["schema"], SCHEMA_GATED)
                self.assertEqual(meta["phase"], "after")
                self.assertEqual(meta["sweep"], 2)
                self.assertEqual(meta["step"], 0)
            c.output.resume = True
            curve = hybrid_pinn.run(c)
            self.assertTrue(all(r["outer_convergence"]["converged"] for r in curve[1:]))
            self.assertTrue(all(r["sweep"] < c.outer_stop.max_sweeps for r in curve[1:]))
            self.assertTrue(all(r["research_advance"] and not r["strict_accepted"] for r in curve))
            paths = [neural_state_root(root / "gated", i) / "state.zip" for i in (1, 2)]
            digests = [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
            self.assertEqual(hybrid_pinn.run(c), curve)
            self.assertEqual(digests, [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths])
            c.data.end_state_index = 3
            self.assertEqual(len(hybrid_pinn.run(c)), 3)
            self.assertEqual(digests, [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths])
            c.output.resume = False
            c.output.fork_from, c.output.fork_after_index = str(root / "gated"), 2
            c.output.dir, c.output.event_dir = str(root / "rejected"), str(root / "rejected_events")
            c.data.end_state_index = 4
            c.outer_stop.max_sweeps = 3
            c.outer_stop.inner_AT2_relative_limit = 0.
            rejected = hybrid_pinn.run(c)
            self.assertEqual(len(rejected), 3)
            self.assertEqual(rejected[-1]["status"], "stagger_budget_exhausted")
            self.assertFalse(rejected[-1]["research_advance"])
            self.assertFalse((root / "rejected/state_0004").exists())
            c.output.resume = True
            with self.assertRaisesRegex(ValueError, "rejected hybrid load"):
                hybrid_pinn.run(c)


if __name__ == "__main__":
    unittest.main()
