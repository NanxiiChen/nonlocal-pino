"""Reference damage-PCG must solve the same equation and reject exhausted work."""

import unittest
import hashlib
import json
from pathlib import Path
import tempfile
from argparse import Namespace

import jax
import numpy as np

from pinn.operators import ExplicitVolumeSparseD
from examples.fracture2d import nonlocal_baseline as direct
from examples.fracture2d.split_reference import damage_pcg, read_trial_seed, trial_archive_path, write_trial_fields
from tests.test_fracture2d_split_horizon import square_cloud
from examples.fracture2d.split_horizon_model import visibility_graph


class DamagePCGReferenceTest(unittest.TestCase):
    def test_anderson_reference_matches_picard_and_restores_secant_history(self):
        from unittest.mock import patch
        from examples.fracture2d import split_reference
        from examples.fracture2d.configs.scale_separation_fine import get_config
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            root = Path(directory)
            cloud = root / "cloud.npz"
            np.savez(cloud, **square_cloud())
            c = get_config()
            c.data.cloud = str(cloud)
            c.data.cloud_sha256 = hashlib.sha256(cloud.read_bytes()).hexdigest()
            c.nl.delta_u, c.nl.delta_d = 0.4, 0.6
            c.geometry.grip_extent, c.geometry.h_grip = 0.6, 0.2
            config = root / "config.py"
            config.write_text("import json\nfrom ml_collections import ConfigDict\n"
                              "def get_config():\n    return ConfigDict(json.loads(" + repr(c.to_json_best_effort()) + "))\n")
            common = dict(config=config, load_step=0.0001, last_load=0.0001,
                          max_sweeps=50, resume=False, fork_from=None, fork_after_index=0,
                          omega=0.5, trial_from=None, anderson_depth=5,
                          anderson_max_backtracks=4, anderson_sufficient_decrease=0.01)
            split_reference.run(Namespace(**common, output=root / "picard", relaxation_policy="picard"))
            original_write = split_reference.atomic_json
            def interrupt_progress(path, payload):
                result = original_write(path, payload)
                if path.name == "progress.json" and payload["sweep"] == 1:
                    raise OSError("manufactured reference interruption")
                return result
            with patch.object(split_reference, "atomic_json", side_effect=interrupt_progress):
                with self.assertRaisesRegex(OSError, "reference interruption"):
                    split_reference.run(Namespace(**common, output=root / "anderson", relaxation_policy="anderson_residual"))
            progress = json.loads((root / "anderson/state_0001/progress.json").read_text())
            self.assertEqual(progress["anderson_history_count"], 1)
            with np.load(trial_archive_path(root / "anderson/state_0001", progress)) as archive:
                self.assertEqual(archive["anderson_x"].shape, (1, 25))
            split_reference.run(Namespace(**{**common, "resume": True}, output=root / "anderson", relaxation_policy="anderson_residual"))
            for method in ("picard", "anderson"):
                report = json.loads((root / method / "state_0001/result.json").read_text())
                self.assertTrue(report["accepted_reference"])
                self.assertTrue(all(report["gates"].values()))
            with np.load(root / "picard/state_0001/committed.npz") as first, np.load(root / "anderson/state_0001/committed.npz") as second:
                for name in ("q", "d", "H"):
                    np.testing.assert_allclose(second[name], first[name], rtol=2e-6, atol=1e-10)

    def test_interrupted_next_generation_preserves_previous_resume_pointer(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = write_trial_fields(root, 1, q=np.zeros(6), d=np.full(3, 0.2), H=np.ones(3))
            # The next field write completes, but progress.json has not changed.
            write_trial_fields(root, 2, q=np.ones(6), d=np.full(3, 0.3), H=np.full(3, 2.))
            path = trial_archive_path(root, first)
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), first["trial_sha256"])
            with np.load(path) as archive:
                np.testing.assert_array_equal(archive["d"], np.full(3, 0.2))
            with self.assertRaisesRegex(ValueError, "basename"):
                trial_archive_path(root, {"trial_file": "../other.npz"})

    def test_trial_seed_checks_parent_and_hash_and_does_not_import_history(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            q, d = np.zeros(6), np.full(3, 0.2)
            np.savez(root / "trial.npz", q=q, d=d, H=np.full(3, 1e9))
            progress = {"model_digest": "model", "parent_sha256": "parent",
                        "load": 0.0042, "trial_sha256": hashlib.sha256(
                            (root / "trial.npz").read_bytes()).hexdigest()}
            (root / "progress.json").write_text(json.dumps(progress))
            kwargs = dict(model_digest="model", parent_sha256="parent",
                          load=0.0042, n_dofs=6, n_material=3)
            actual_q, actual_d = read_trial_seed(root, **kwargs)
            np.testing.assert_array_equal(actual_q, q)
            np.testing.assert_array_equal(actual_d, d)
            with self.assertRaisesRegex(ValueError, "parent/digest"):
                read_trial_seed(root, **{**kwargs, "parent_sha256": "other"})
            np.savez(root / "trial.npz", q=q + 1, d=d)
            with self.assertRaisesRegex(ValueError, "parent/digest"):
                read_trial_seed(root, **kwargs)

    def test_matches_independent_direct_solve_and_fails_loudly(self):
        jax.config.update("jax_enable_x64", True)
        cloud = square_cloud()
        _, _, edges, _ = visibility_graph(cloud["coordinates"], cloud["triangles"], 0.6, 1e-12)
        operator = ExplicitVolumeSparseD(cloud["coordinates"], cloud["volumes"], edges, 0.6, "cubic")
        H = np.linspace(0.01, 50, len(cloud["coordinates"]))
        expected = direct.solve_fixed_history_damage(operator, 2.7, 0.024, H)
        actual, info = damage_pcg(operator.to_scipy(), cloud["volumes"], 2.7 / 0.024,
                                  2.7 * 0.024, H, np.zeros_like(H), rtol=1e-12, atol=1e-14, maxiter=100)
        np.testing.assert_allclose(actual, expected.damage, rtol=1e-11, atol=1e-13)
        self.assertLessEqual(info["volume_residual_norm"], info["limit"] * 1.001)
        with self.assertRaisesRegex(RuntimeError, "explicit check"):
            damage_pcg(operator.to_scipy(), cloud["volumes"], 2.7 / 0.024, 2.7 * 0.024,
                       H, np.zeros_like(H), rtol=1e-15, atol=0, maxiter=1)


if __name__ == "__main__":
    unittest.main()
