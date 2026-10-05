"""Protect warning provenance and exclusion of rejected reference fields."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zipfile
from argparse import Namespace
from unittest.mock import patch

import jax

import numpy as np

from examples.fracture2d.configs.split_pinn_fine import get_config
from examples.fracture2d.split_pinn import classify_report, comparison_reference, resume_identity, fork_checkpoint_identity, neural_state_root


class SplitPINNPolicyTest(unittest.TestCase):
    def test_fork_preserves_physics_and_rejects_cycles(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stored = dict(model_digest="model", reference_sha256="ref", parent_sha256="parent", index=3,
                          config={"arch": {"width": 8}, "physical": {"Gc": 2.7}, "nl": {"delta_u": 0.008},
                                  "geometry": {"grip_extent": 0.048}, "data": {"load_step": 0.0001}})
            path = root / "state.zip"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("metadata.json", json.dumps({"identity": stored}))
            expected = json.loads(json.dumps(stored))
            expected["config"]["output"] = {"fork_from": "source"}
            self.assertEqual(fork_checkpoint_identity(path, expected), stored)
            expected["config"]["physical"]["Gc"] = 3.
            with self.assertRaisesRegex(ValueError, "physical differs"):
                fork_checkpoint_identity(path, expected)
            (root / "identity.json").write_text(json.dumps({"config": {"output": {
                "fork_from": str(root), "fork_after_index": 3}}}))
            with self.assertRaisesRegex(ValueError, "cyclic"):
                neural_state_root(root, 1)

    def test_endpoint_extension_preserves_numerical_identity(self):
        old = {"config": {"data": {"end_state_index": 2}, "lr": 1e-4}}
        extended = {"config": {"data": {"end_state_index": 4}, "lr": 1e-4}}
        resume_identity(old, extended)
        self.assertEqual(old["config"]["data"]["end_state_index"], 2)
        with self.assertRaisesRegex(ValueError, "endpoint extension only"):
            resume_identity(old, {"config": {"data": {"end_state_index": 4}, "lr": 2e-4}})
        with self.assertRaisesRegex(ValueError, "endpoint extension only"):
            resume_identity(extended, old)

    @unittest.skipUnless(any(device.platform == "gpu" for device in jax.devices()), "GPU integration")
    def test_small_neural_loading_and_completed_resume(self):
        from examples.fracture2d.split_reference import run as run_reference
        from examples.fracture2d.split_pinn import run as run_pinn
        from tests.test_fracture2d_split_horizon import square_cloud
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            root = Path(directory)
            c = get_config()
            cloud = root / "cloud.npz"
            np.savez(cloud, **square_cloud())
            c.data.cloud = str(cloud)
            c.data.cloud_sha256 = hashlib.sha256(cloud.read_bytes()).hexdigest()
            c.nl.delta_u, c.nl.delta_d = 0.4, 0.6
            c.geometry.grip_extent, c.geometry.h_grip = 0.6, 0.2
            c.arch.hidden_dim, c.arch.num_layers = 8, 1
            c.train.mechanics_objective = "energy"
            c.train.operator_mode = "analytic_padded_damage"
            c.stagger.sweeps = 1
            c.stagger.mechanics_steps = c.stagger.damage_steps = 2
            c.output.log_every = 2
            c.reporting.reaction_relative_limit = 10.
            c.reporting.damage_rmse_limit = 0.1
            c.output.dir = str(root / "pinn")
            c.data.reference_root = str(root / "reference")
            config_path = root / "config.py"
            config_path.write_text("import json\nfrom ml_collections import ConfigDict\n"
                                   "def get_config():\n    return ConfigDict(json.loads(" + repr(c.to_json_best_effort()) + "))\n")
            run_reference(Namespace(config=config_path, output=root / "reference", load_step=0.0001,
                                    last_load=0.0003, max_sweeps=50, resume=False,
                                    fork_from=None, fork_after_index=0, omega=0.5, trial_from=None))
            from examples.fracture2d import split_pinn
            original_write = split_pinn.atomic_json
            def interrupt_report(path, payload):
                if path.name == "sweep_0001.json" and path.parent.name == "state_0001":
                    raise OSError("manufactured report interruption")
                return original_write(path, payload)
            with patch.object(split_pinn, "atomic_json", side_effect=interrupt_report):
                with self.assertRaisesRegex(OSError, "report interruption"):
                    run_pinn(c)
            c.output.resume = True
            curve = run_pinn(c)
            self.assertEqual(len(curve), 2)
            self.assertTrue(all(row["research_advance"] for row in curve))
            self.assertFalse(curve[0]["strict_accepted"])
            self.assertTrue(curve[1]["inherited_warning"])
            with np.load(root / "pinn/state_0002/fields_0001.npz") as fields:
                self.assertTrue(np.all(np.isfinite(fields["d"])))
            c.output.resume = True
            restored = run_pinn(c)
            self.assertEqual(restored, curve)
            previous_digest = hashlib.sha256((root / "pinn/state_0002/state.zip").read_bytes()).hexdigest()
            c.data.end_state_index = 3
            extended = run_pinn(c)
            self.assertEqual(len(extended), 3)
            self.assertTrue(extended[-1]["inherited_warning"])
            self.assertEqual(hashlib.sha256((root / "pinn/state_0002/state.zip").read_bytes()).hexdigest(), previous_digest)
            self.assertTrue((root / "pinn/reaction_curve.png").is_file())

    def test_secondary_warning_can_advance_but_primary_failure_cannot(self):
        limits = get_config().reporting
        report = dict(damage_rmse=0.001, reaction_relative=0.02, mechanics_loss=0.1,
                      coupled_AT2_relative=0.1, damage_change=0.01,
                      displacement_relative=0.3, healing=0., damage_min=0.,
                      damage_max=0.99, drift_count=2, history_decrease=0.)
        result = classify_report(report, limits, inherited_warning=False)
        self.assertTrue(result["research_advance"])
        self.assertFalse(result["strict_accepted"])
        self.assertEqual(result["status"], "warning")
        self.assertEqual(result["checks"]["drift_count"], {"value": 2., "limit": 0., "pass": False})
        failed = classify_report({**report, "damage_rmse": 0.1}, limits, inherited_warning=False)
        self.assertFalse(failed["research_advance"])
        clean = {**report, "mechanics_loss": 0., "coupled_AT2_relative": 0.,
                 "damage_change": 0., "displacement_relative": 0., "drift_count": 0}
        self.assertTrue(classify_report(clean, limits, inherited_warning=False)["strict_accepted"])
        self.assertFalse(classify_report(clean, limits, inherited_warning=True)["strict_accepted"])
        with self.assertRaises(FloatingPointError):
            classify_report({**report, "mechanics_loss": np.nan}, limits, inherited_warning=False)

    def test_comparison_rejects_unaccepted_reference_and_mutated_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "policy.json").write_text("{}")
            state = root / "state_0001"
            state.mkdir()
            np.savez(state / "committed.npz", q=np.zeros(6), d=np.zeros(3), H=np.ones(3))
            digest = hashlib.sha256((state / "committed.npz").read_bytes()).hexdigest()
            report = dict(accepted_reference=True, model_digest="model", load=0.0001,
                          committed_sha256=digest, reaction=1.)
            (state / "result.json").write_text(json.dumps(report))
            q, d, reaction, actual_digest = comparison_reference(root, 1, "model", 6, 3, 0.0001)
            self.assertEqual(actual_digest, digest)
            self.assertEqual(reaction, 1.)
            self.assertEqual(d.shape, (3,))
            (state / "result.json").write_text(json.dumps({**report, "accepted_reference": False}))
            with self.assertRaisesRegex(ValueError, "matching strict"):
                comparison_reference(root, 1, "model", 6, 3, 0.0001)
            (state / "result.json").write_text(json.dumps(report))
            np.savez(state / "committed.npz", q=q + 1, d=d)
            with self.assertRaisesRegex(ValueError, "matching strict"):
                comparison_reference(root, 1, "model", 6, 3, 0.0001)


if __name__ == "__main__":
    unittest.main()
