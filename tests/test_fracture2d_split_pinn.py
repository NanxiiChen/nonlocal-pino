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
            stored["reference_provenance"] = {"status": "warning", "policy_sha256": "bound-policy"}
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("metadata.json", json.dumps({"identity": stored}))
            expected = json.loads(json.dumps(stored))
            self.assertEqual(fork_checkpoint_identity(path, expected), stored)
            expected["reference_provenance"]["status"] = "strict"
            with self.assertRaisesRegex(ValueError, "reference provenance"):
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
            c.reporting.comparison_gate = False
            c.stagger.accept_on_first_passing_report = False
            c.reporting.reaction_relative_limit = 0.
            c.reporting.damage_rmse_limit = 0.
            c.output.dir = str(root / "pinn")
            c.data.reference_root = str(root / "reference")
            config_path = root / "config.py"
            config_path.write_text("import json\nfrom ml_collections import ConfigDict\n"
                                   "def get_config():\n    return ConfigDict(json.loads(" + repr(c.to_json_best_effort()) + "))\n")
            run_reference(Namespace(config=config_path, output=root / "reference", load_step=0.0001,
                                    last_load=0.0003, max_sweeps=50, resume=False,
                                    fork_from=None, fork_after_index=0, omega=0.5, trial_from=None))
            from examples.fracture2d.warning_reference import run as run_warning
            run_warning(Namespace(config=config_path, source_root=root / "reference", output=root / "warning",
                fork_after_index=1, last_load=.0003, sweeps_per_load=3, omega=.5,
                seed_state=None, seed_sha256=None, damage_backend="cpu", resume=False))
            c.data.reference_root = str(root / "warning")
            c.data.reference_policy = "user_accepted_warning"
            c.data.reference_policy_sha256 = hashlib.sha256((root / "warning/policy.json").read_bytes()).hexdigest()
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
            self.assertTrue(all(not row["comparison_checks_pass"] for row in curve))
            self.assertFalse(curve[0]["strict_accepted"])
            self.assertTrue(curve[1]["inherited_warning"])
            self.assertEqual(curve[0]["reference_provenance"]["status"], "strict")
            self.assertEqual(curve[1]["reference_provenance"]["status"], "warning")
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

    def test_accepted_warning_requires_bound_policy_parent_and_preserves_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, warning = root / "strict", root / "warning"
            (source / "state_0001").mkdir(parents=True)
            (warning / "state_0002").mkdir(parents=True)
            (source / "policy.json").write_text("{}")
            parent = source / "state_0001/committed.npz"
            np.savez(parent, q=np.zeros(6), d=np.zeros(3), H=np.zeros(3))
            parent_sha = hashlib.sha256(parent.read_bytes()).hexdigest()
            (source / "state_0001/result.json").write_text(json.dumps(dict(
                accepted_reference=True, model_digest="model", load=.0001,
                committed_sha256=parent_sha, reaction=1.)))
            policy = dict(schema="fracture2d-split-warning-v1", diagnostic_only=True,
                model_digest="model", fork_from=str(source), fork_after_index=1)
            (warning / "policy.json").write_text(json.dumps(policy))
            policy_sha = hashlib.sha256((warning / "policy.json").read_bytes()).hexdigest()
            path = warning / "state_0002/warning.npz"
            np.savez(path, q=np.zeros(6), d=np.full(3,.1), H=np.ones(3))
            report = dict(accepted_reference=False, model_digest="model", load=.0002,
                warning_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), parent_sha256=parent_sha,
                diagnostic_only=True, warning_usable=True, strict_accepted=False,
                field_equations_validated=False, inherited_warning=False, reaction=2.,
                hard_gates=dict(finite=True,damage_range=True,history_monotonic=True),
                warning_failures=["coupled_damage"])
            report_path = warning / "state_0002/result.json"
            report_path.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError,"strict reference"):
                comparison_reference(warning,2,"model",6,3,.0002)
            kwargs = dict(reference_policy="user_accepted_warning", policy_sha256=policy_sha, return_metadata=True)
            with self.assertRaisesRegex(ValueError,"policy SHA"):
                comparison_reference(warning,2,"model",6,3,.0002,**{**kwargs,"policy_sha256":"wrong"})
            *_,metadata = comparison_reference(warning,2,"model",6,3,.0002,**kwargs)
            self.assertEqual(metadata["status"],"warning")
            self.assertTrue(metadata["diagnostic_only"])
            self.assertEqual(metadata["warning_failures"],["coupled_damage"])
            self.assertEqual(comparison_reference(warning,1,"model",6,3,.0001,**kwargs)[-1]["status"],"strict")
            self.assertFalse(json.loads(report_path.read_text())["accepted_reference"])
            report_path.write_text(json.dumps({**report,"parent_sha256":"wrong"}))
            with self.assertRaisesRegex(ValueError,"parent SHA"):
                comparison_reference(warning,2,"model",6,3,.0002,**kwargs)
            report_path.write_text(json.dumps({**report,"hard_gates":{}}))
            with self.assertRaisesRegex(ValueError,"status/ancestry"):
                comparison_reference(warning,2,"model",6,3,.0002,**kwargs)
            report_path.write_text(json.dumps(report))
            np.savez(path,q=np.ones(6),d=np.full(3,.1))
            with self.assertRaisesRegex(ValueError,"field digest"):
                comparison_reference(warning,2,"model",6,3,.0002,**kwargs)

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

    def test_explicit_full_warning_mode_keeps_bad_reference_checks_without_rejecting(self):
        limits = get_config().reporting
        report = dict(damage_rmse=.010068, reaction_relative=.072117, mechanics_loss=.0349,
            coupled_AT2_relative=.0563, damage_change=.0137, displacement_relative=.0145,
            healing=.00003, damage_min=0., damage_max=.59, drift_count=0, history_decrease=0.)
        self.assertFalse(classify_report(report,limits,inherited_warning=True)["research_advance"])
        limits.comparison_gate = False
        warning = classify_report(report,limits,inherited_warning=True)
        self.assertTrue(warning["research_advance"])
        self.assertFalse(warning["comparison_checks_pass"])
        self.assertFalse(warning["comparison_gate_enforced"])
        self.assertFalse(warning["strict_accepted"])
        self.assertEqual(warning["status"],"warning")
        self.assertIn("reaction_relative",warning["failed_checks"])
        self.assertIn("damage_rmse",warning["failed_checks"])
        with self.assertRaises(FloatingPointError):
            classify_report({**report,"reaction":np.nan},limits,inherited_warning=True)
        bad_limits = get_config().reporting
        bad_limits.comparison_gate = "false"
        with self.assertRaisesRegex(ValueError,"comparison_gate must be boolean"):
            classify_report(report,bad_limits,inherited_warning=True)

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
