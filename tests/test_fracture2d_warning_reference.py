"""Warning diagnostics must retain strict failures and never enter strict readers."""

from argparse import Namespace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from examples.fracture2d import split_reference, warning_reference
from examples.fracture2d.configs.scale_separation_fine import get_config
from examples.fracture2d.split_pinn import comparison_reference
from tests.test_fracture2d_split_horizon import square_cloud


class WarningReferenceTest(unittest.TestCase):
    def test_soft_failures_warn_but_invalid_range_and_history_stop(self):
        limits = dict(range_atol=1e-10,healing_atol=1e-3,coupled_AT2_relative_inf_limit=1e-6,
            damage_change_atol=1e-5,q_change_relative_limit=1e-5,H_change_relative_limit=1e-5,
            reaction_change_relative_limit=1e-5)
        d,H,q,rho = np.array([.4]),np.array([1.]),np.array([2.,0.]),np.array([1.])
        kwargs = dict(floor_d=np.array([.5]),floor_H=np.array([0.]),previous_d=d.copy(),
            load=1.,n_material=1,scale=2.,previous=None,reaction=1.,limits=limits)
        original,hard,*_ = warning_reference.gates(d,H,q,rho,**kwargs)
        self.assertFalse(original["healing"])
        self.assertFalse(original["drift"])
        self.assertFalse(original["coupled_damage"])
        self.assertTrue(all(hard.values()))
        self.assertFalse(warning_reference.gates(np.array([1.1]),H,q,rho,**kwargs)[1]["damage_range"])
        kwargs["floor_H"] = np.array([2.])
        self.assertFalse(warning_reference.gates(d,H,q,rho,**kwargs)[1]["history_monotonic"])

    def test_warning_ancestry_resume_and_strict_comparator_rejection(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            root = Path(directory)
            cloud = root/"cloud.npz"
            np.savez(cloud,**square_cloud())
            c = get_config()
            c.data.cloud,c.data.cloud_sha256 = str(cloud),hashlib.sha256(cloud.read_bytes()).hexdigest()
            c.nl.delta_u,c.nl.delta_d = .4,.6
            c.geometry.grip_extent,c.geometry.h_grip = .6,.2
            config = root/"config.py"
            config.write_text("import json\nfrom ml_collections import ConfigDict\ndef get_config():\n"
                              "    return ConfigDict(json.loads("+repr(c.to_json_best_effort())+"))\n")
            source = root/"strict"
            common = dict(config=config,output=source,load_step=1e-4,last_load=1e-4,
                max_sweeps=50,resume=False,fork_from=None,fork_after_index=0,omega=.5,trial_from=None)
            split_reference.run(Namespace(**common))
            kwargs = dict(config=config,source_root=source,output=root/"warning",fork_after_index=1,
                last_load=3e-4,sweeps_per_load=3,omega=.5,seed_state=None,seed_sha256=None,
                damage_backend="cpu",resume=False)
            original_write = warning_reference.atomic_json
            def interrupt_endpoint(path,payload):
                if path.name=="result.json" and path.parent.name=="state_0002":
                    raise OSError("manufactured warning publication interruption")
                return original_write(path,payload)
            with patch.object(warning_reference,"atomic_json",side_effect=interrupt_endpoint):
                with self.assertRaisesRegex(OSError,"publication interruption"):
                    warning_reference.run(Namespace(**kwargs))
            self.assertTrue((root/"warning/state_0002/warning.npz").is_file())
            before = hashlib.sha256((root/"warning/state_0002/warning.npz").read_bytes()).hexdigest()
            result = warning_reference.run(Namespace(**{**kwargs,"resume":True}))
            self.assertTrue(result["complete"])
            self.assertTrue(result["diagnostic_only"])
            self.assertFalse(result["strict_accepted"])
            self.assertEqual(hashlib.sha256((root/"warning/state_0002/warning.npz").read_bytes()).hexdigest(),before)
            second = json.loads((root/"warning/state_0002/result.json").read_text())
            third = json.loads((root/"warning/state_0003/result.json").read_text())
            self.assertFalse(second["inherited_warning"])
            self.assertTrue(third["inherited_warning"])
            self.assertEqual(third["parent_sha256"],second["warning_sha256"])
            self.assertFalse((root/"warning/state_0002/committed.npz").exists())
            with self.assertRaisesRegex(ValueError,"strict reference"):
                comparison_reference(root/"warning",2,second["model_digest"],110,25,2e-4)
