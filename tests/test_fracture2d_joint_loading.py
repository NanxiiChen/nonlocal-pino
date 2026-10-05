"""Research continuation must not promote unaccepted trial history."""

from argparse import Namespace
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from examples.fracture2d.joint_loading import run


class JointLoadingContractTest(unittest.TestCase):
    def setup_case(self, root):
        parent = root / "parent"
        parent.mkdir()
        (parent / "fields.npz").write_bytes(b"committed parent field identity")
        (parent / "result.json").write_text(json.dumps({
            "physical_checks_pass": True, "optimizer_success": True, "load": 0.0065}))
        (parent / "policy.json").write_text(json.dumps({
            "base": "bound-base", "source": "bound-source", "source_index": 1,
            "parent_digest": "a" * 64}))
        return Namespace(parent=parent, output=root / "run", last_load=0.0067,
            load_step=0.0001, max_evaluations=20, lsmr_maxiter=30, lsmr_tolerance=1e-6,
            damage_weight=10.0, extra_q_units=0, mechanics_weighting="initial_outliers",
            max_reweights=2, first_trial=None, resume=False)

    def fake_solver(self, success_sequence, calls):
        def execute(command, **kwargs):
            calls.append(command)
            def value(flag):
                return command[command.index(flag) + 1]
            parent = Path(value("--joint-parent"))
            output = Path(value("--output"))
            output.mkdir()
            load = float(value("--load"))
            (output / "fields.npz").write_bytes(str(output).encode())
            (output / "policy.json").write_text(json.dumps({
                "load": load, "joint_parent": str(parent), "base": "bound-base",
                "source": "bound-source", "source_index": 1, "parent_digest": "a" * 64,
                "source_field_sha256": hashlib.sha256((parent / "fields.npz").read_bytes()).hexdigest()}))
            success = success_sequence[len(calls) - 1]
            (output / "result.json").write_text(json.dumps({
                "physical_checks_pass": success, "optimizer_success": success, "load": load}))
        return execute

    def test_rejected_refinements_keep_original_committed_parent(self):
        with TemporaryDirectory() as temporary:
            args = self.setup_case(Path(temporary))
            calls = []
            with patch("examples.fracture2d.joint_loading.subprocess.run",
                       side_effect=self.fake_solver([False, False, False], calls)):
                run(args)
            self.assertEqual(len(calls), 3)
            for command in calls:
                self.assertEqual(command[command.index("--joint-parent") + 1], str(args.parent))
                self.assertAlmostEqual(float(command[command.index("--load") + 1]), 0.0066)
            result = json.loads((args.output / "result.json").read_text())
            self.assertFalse(result["complete"])
            self.assertEqual(result["last_usable_parent"], str(args.parent))

    def test_only_passing_refinement_advances_and_resume_checks_parent_digest(self):
        with TemporaryDirectory() as temporary:
            args = self.setup_case(Path(temporary))
            calls = []
            with patch("examples.fracture2d.joint_loading.subprocess.run",
                       side_effect=self.fake_solver([False, True, True], calls)):
                run(args)
            self.assertEqual(len(calls), 3)
            refined = args.output / "state_0001_reweight_01"
            self.assertEqual(calls[2][calls[2].index("--joint-parent") + 1], str(refined))
            self.assertTrue(json.loads((args.output / "result.json").read_text())["complete"])
            (args.parent / "fields.npz").write_bytes(b"changed committed parent")
            args.resume = True
            with patch("examples.fracture2d.joint_loading.subprocess.run") as execute:
                with self.assertRaisesRegex(ValueError, "digest changed"):
                    run(args)
                execute.assert_not_called()


if __name__ == "__main__":
    unittest.main()
