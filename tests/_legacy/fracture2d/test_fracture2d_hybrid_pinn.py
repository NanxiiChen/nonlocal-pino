"""Hybrid mechanics/history accuracy and damage-phase recovery contracts."""

from argparse import Namespace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import jax
import jax.numpy as jnp
import numpy as np

from examples.fracture2d._legacy.configs.hybrid_pinn_short_damage import get_config
from examples.fracture2d._legacy.hybrid_pinn import SCHEMA, run, validate_config


class HybridPINNTest(unittest.TestCase):
    def test_explicit_solver_and_controls(self):
        c = get_config()
        self.assertEqual(validate_config(c), [1, 2])
        self.assertNotIn("mechanics_steps", c.stagger)
        self.assertNotIn("mechanics_objective", c.train)
        c.train.mechanics_solver = "pinn"
        with self.assertRaisesRegex(ValueError, "direct mechanics"):
            validate_config(c)
        c = get_config()
        c.validation.cg_rtol = c.validation.cg_atol = 0.
        with self.assertRaisesRegex(ValueError, "cannot both be zero"):
            validate_config(c)
        c = get_config()
        c.output.fork_from = "full-neural-history"
        with self.assertRaisesRegex(ValueError, "explicit nonempty prefix"):
            validate_config(c)
        c = get_config()
        c.stagger.accept_on_first_passing_report = True
        with self.assertRaisesRegex(ValueError, "external reference"):
            validate_config(c)

    @unittest.skipUnless(any(d.platform == "gpu" for d in jax.devices()), "GPU integration")
    def test_own_history_direct_force_damage_resume_and_extension(self):
        from examples.fracture2d._legacy import hybrid_pinn
        from examples.fracture2d._legacy.fp64 import FP64A2Backend
        from examples.fracture2d._legacy.losses import FractureResidual
        from examples.fracture2d._legacy.split_horizon_model import build_model, read_cloud
        from examples.fracture2d._legacy.split_reference import run as run_reference
        from tests._legacy.fracture2d.test_fracture2d_split_horizon import square_cloud
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            root = Path(directory)
            c = get_config()
            cloud = root / "cloud.npz"
            np.savez(cloud, **square_cloud())
            c.data.cloud = str(cloud)
            c.data.cloud_sha256 = hashlib.sha256(cloud.read_bytes()).hexdigest()
            c.nl.delta_u, c.nl.delta_d = .4, .6
            c.geometry.grip_extent, c.geometry.h_grip = .6, .2
            c.arch.hidden_dim, c.arch.num_layers = 8, 1
            c.stagger.sweeps, c.stagger.damage_steps = 2, 2
            c.output.log_every = 2
            c.output.dir, c.output.event_dir = str(root / "hybrid"), str(root / "events")
            c.data.reference_root, c.data.reference_policy = str(root / "reference"), "strict"
            config = root / "config.py"
            config.write_text("import json\nfrom ml_collections import ConfigDict\n"
                "def get_config():\n    return ConfigDict(json.loads(" + repr(c.to_json_best_effort()) + "))\n")
            run_reference(Namespace(config=config, output=root / "reference", load_step=.0001,
                last_load=.0003, max_sweeps=50, resume=False, fork_from=None,
                fork_after_index=0, omega=.5, trial_from=None))
            original_write = hybrid_pinn.write_checkpoint
            def interrupt_after_damage(path, identity, sweep, phase, step, state, **kwargs):
                result = original_write(path, identity, sweep, phase, step, state, **kwargs)
                if path.name == "progress.zip" and phase == "after":
                    raise OSError("manufactured after-damage interruption")
                return result
            with patch.object(hybrid_pinn, "write_checkpoint", side_effect=interrupt_after_damage):
                with self.assertRaisesRegex(OSError, "after-damage interruption"):
                    run(c)
            with zipfile.ZipFile(root / "hybrid/state_0001/progress.zip") as archive:
                metadata = json.loads(archive.read("metadata.json"))
                self.assertEqual(metadata["schema"], SCHEMA)
                self.assertEqual(metadata["phase"], "after")
            c.output.resume = True
            curve = run(c)
            self.assertEqual(len(curve), 2)
            self.assertTrue(all(r["mechanics"]["converged"] and r["research_advance"] for r in curve))
            self.assertTrue(all(r["mechanics_solver"] == "direct" and not r["strict_accepted"] for r in curve))
            model = build_model(read_cloud(c.data.cloud, c.data.cloud_sha256),
                delta_u=c.nl.delta_u, delta_d=c.nl.delta_d, grip_extent=c.geometry.grip_extent,
                h_grip=c.geometry.h_grip, visibility_tolerance=c.nl.visibility_tolerance,
                E=c.physical.E, nu=c.physical.nu, k_res=c.physical.k_res,
                rank_rtol=c.physical.rank_rtol, Gc=c.physical.Gc, l0=c.physical.l0)
            backend = FP64A2Backend(model.static, model.constraints, jax.devices()[0])
            residual = FractureResidual(model, backend)
            floor = jnp.zeros(model.n_material, jnp.float64)
            for index in (1, 2):
                for sweep in (1, 2):
                    with np.load(root / f"hybrid/state_{index:04d}/fields_{sweep:04d}.npz") as f:
                        q, d, H, H_used, image, previous = (f[n] for n in
                            ("q", "d", "H", "H_used", "d_image", "d_previous"))
                    load = jnp.asarray(index*.0001, jnp.float64)
                    evaluated = residual.evaluate(jnp.asarray(q), jnp.asarray(d), None, floor)
                    np.testing.assert_allclose(H, evaluated.trial_history, rtol=1e-12, atol=1e-12)
                    mechanical = json.loads((root / f"hybrid/state_{index:04d}/mechanics_{sweep:04d}_after.json").read_text())
                    self.assertLessEqual(float(jnp.linalg.norm(evaluated.mechanics)),
                                         mechanical["convergence_limit"] + 1e-10)
                    np.testing.assert_allclose(d, previous+c.stagger.omega*(image-previous), rtol=1e-12, atol=1e-15)
                    before = backend.solve_strict(previous, load, c.validation.cg_rtol,
                        c.validation.cg_atol, c.validation.cg_maxiter)
                    history_before = residual.evaluate(before.displacement, jnp.asarray(previous), None, floor).trial_history
                    np.testing.assert_allclose(H_used, history_before, rtol=1e-8, atol=1e-10)
                    self.assertTrue(np.all(H >= np.asarray(floor)))
                floor = jnp.asarray(H)
            digests = [hashlib.sha256((root / f"hybrid/state_{i:04d}/state.zip").read_bytes()).hexdigest() for i in (1, 2)]
            self.assertEqual(run(c), curve)
            self.assertEqual(digests, [hashlib.sha256((root / f"hybrid/state_{i:04d}/state.zip").read_bytes()).hexdigest() for i in (1, 2)])
            c.data.end_state_index = 3
            extended = run(c)
            self.assertEqual(len(extended), 3)
            self.assertEqual(digests[1], hashlib.sha256((root / "hybrid/state_0002/state.zip").read_bytes()).hexdigest())
            timing = json.loads((root / "hybrid/state_0003/timing.json").read_text())
            self.assertEqual(timing["phase_updates"], {"before": 0, "damage": 4, "after": 0})
            c.validation.cg_rtol = 1e-8
            with self.assertRaisesRegex(ValueError, "endpoint extension only"):
                run(c)


if __name__ == "__main__":
    unittest.main()
