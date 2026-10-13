"""Loading-history identity and stable-but-inaccurate stopping regressions."""
from argparse import Namespace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import jax
import numpy as np

from examples.fracture2d.configs.split_pinn_adaptive import get_config
from examples.fracture2d.configs.split_pinn_accelerated import get_config as accelerated_config
from examples.fracture2d.configs.split_pinn_fine import get_config as fine_config
from examples.fracture2d.inner_stopping import StopWindow, validate_controls
from examples.fracture2d.split_loading import reference_indices, check_fork_loading
from examples.fracture2d.split_pinn import resume_identity


class LoadingPlanTest(unittest.TestCase):
    def test_user_example_maps_30_ordinals_to_direct_states(self):
        c = accelerated_config()
        refs = reference_indices(c.data)
        self.assertEqual(len(refs), 30)
        self.assertEqual(refs[:6], [10,20,30,40,50,51])
        self.assertEqual(refs[-1], 75)
        self.assertAlmostEqual(refs[0]*c.data.load_step, .001)
        self.assertAlmostEqual(refs[4]*c.data.load_step, .005)
        self.assertAlmostEqual(refs[-1]*c.data.load_step, .0075)

    def test_invalid_order_and_endpoint_fail(self):
        for values in ([1,1],[3,2],[True,3],[1,2.5],[-1,2]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                reference_indices(dict(load_step=.0001,end_state_index=2,reference_indices=values))
        with self.assertRaisesRegex(ValueError,'shorter'):
            reference_indices(dict(load_step=.0001,end_state_index=3,reference_indices=[1,2]))
        self.assertEqual(reference_indices(dict(load_step=.0001,end_state_index=3)),[1,2,3])

    def test_fork_cannot_rewrite_parent_history_and_resume_cannot_rewrite_schedule(self):
        old=dict(load_step=.0001,end_state_index=3)
        new={**old,'reference_indices':[1,2,10]}
        check_fork_loading(old,new,2)
        with self.assertRaisesRegex(ValueError,'inherited loading'):
            check_fork_loading(old,new,3)
        a={'config':{'data':{**new,'end_state_index':2}}}
        b={'config':{'data':{**new,'end_state_index':3}}}
        resume_identity(a,b)
        b['config']['data']['reference_indices']=[1,2,11]
        with self.assertRaisesRegex(ValueError,'endpoint extension'):
            resume_identity(a,b)


class InnerStoppingTest(unittest.TestCase):
    def feed(self, window, residual, *, spike=None, physical=True):
        result=None
        for k in range(1,7):
            result=window.observe(k*100,objective=.3 if k!=spike else .4,
                residual=residual,field_change=None if k==1 else 0.,reaction=10.,physical_pass=physical)
            if k<6:
                self.assertIsNone(result)
        return result

    def test_five_intervals_and_residual_accuracy_distinguish_convergence_from_stall(self):
        c=get_config().inner_stop
        validate_controls(c,2000)
        self.assertEqual(self.feed(StopWindow(c,'before',200),.005),'converged')
        self.assertEqual(self.feed(StopWindow(c,'before',200),.05),'stagnation_warning')
        c.allow_stagnation_warning=False
        self.assertIsNone(self.feed(StopWindow(c,'before',200),.05))

    def test_window_spike_invalid_fields_and_training_minimum_do_not_stop(self):
        c=get_config().inner_stop
        self.assertIsNone(self.feed(StopWindow(c,'before',200),.005,spike=4))
        self.assertIsNone(self.feed(StopWindow(c,'before',200),.005,physical=False))
        self.assertIsNone(self.feed(StopWindow(c,'before',2000),.005))
        with self.assertRaisesRegex(ValueError,'minimum exceeds'):
            validate_controls(c,100)

    def test_damage_uses_its_own_accuracy_target_and_nonfinite_is_fatal(self):
        c=get_config().inner_stop
        self.assertEqual(self.feed(StopWindow(c,'damage',200),.005),'stagnation_warning')
        window=StopWindow(c,'damage',200)
        with self.assertRaises(FloatingPointError):
            window.observe(100,objective=np.nan,residual=.01,field_change=None,reaction=0.,physical_pass=True)

    @unittest.skipUnless(any(d.platform=='gpu' for d in jax.devices()),'GPU integration')
    def test_nonuniform_adaptive_loading_interrupt_resume_and_extension(self):
        from examples.fracture2d import split_pinn, split_reference
        from tests.test_fracture2d_split_horizon import square_cloud
        with tempfile.TemporaryDirectory(dir='/tmp/opencode') as directory:
            root=Path(directory);c=fine_config()
            cloud=root/'cloud.npz';np.savez(cloud,**square_cloud())
            c.data.cloud=str(cloud);c.data.cloud_sha256=hashlib.sha256(cloud.read_bytes()).hexdigest()
            c.nl.delta_u,c.nl.delta_d=.4,.6
            c.geometry.grip_extent,c.geometry.h_grip=.6,.2
            c.arch.hidden_dim,c.arch.num_layers=8,1
            c.data.reference_root=str(root/'direct')
            c.data.reference_indices=[1,3,5]
            c.data.end_state_index=2
            c.train.operator_mode='analytic_padded_damage'
            c.stagger.sweeps=1;c.stagger.mechanics_steps=2;c.stagger.damage_steps=2
            c.stagger.accept_on_first_passing_report=False
            c.inner_stop=get_config().inner_stop
            c.inner_stop.check_every=1;c.inner_stop.min_updates=1;c.inner_stop.first_load_min_updates=1
            c.inner_stop.stable_checks=1
            for key in ('u_change_relative','d_change_atol','reaction_atol','reaction_rtol',
                        'objective_atol','objective_rtol','residual_atol','residual_rtol',
                        'mechanics_limit','damage_limit'):
                c.inner_stop[key]=1e6  # Tiny integration fixture, not a physical accuracy test.
            c.reporting.reaction_relative_limit=10.;c.reporting.damage_rmse_limit=.1
            c.output.dir=str(root/'pinn');c.output.log_every=2
            config=root/'config.py'
            config.write_text('import json\nfrom ml_collections import ConfigDict\ndef get_config():\n'
                '    return ConfigDict(json.loads('+repr(c.to_json_best_effort())+'))\n')
            split_reference.run(Namespace(config=config,output=root/'direct',load_step=.0001,
                last_load=.0005,max_sweeps=50,resume=False,fork_from=None,fork_after_index=0,
                omega=.5,trial_from=None))
            write=split_pinn.atomic_json
            def interrupt(path,payload):
                if path.name=='sweep_0001.json' and path.parent.name=='state_0001':
                    raise OSError('manufactured adaptive report interruption')
                return write(path,payload)
            with patch.object(split_pinn,'atomic_json',side_effect=interrupt):
                with self.assertRaisesRegex(OSError,'adaptive report interruption'):
                    split_pinn.run(c)
            c.output.resume=True
            rows=split_pinn.run(c)
            self.assertEqual(rows[1]['reference_index'],3)
            self.assertAlmostEqual(rows[1]['load'],.0003)
            self.assertAlmostEqual(rows[1]['load_increment'],.0002)
            self.assertTrue(rows[1]['research_advance'])
            self.assertEqual([r['phase'] for r in rows[0]['inner_stops']],['before','damage','after'])
            self.assertTrue(rows[0]['inner_stops'][1]['monitors_unrelaxed_image'])
            digest=hashlib.sha256((root/'pinn/state_0002/state.zip').read_bytes()).hexdigest()
            self.assertEqual(split_pinn.run(c),rows)
            c.data.end_state_index=3
            extended=split_pinn.run(c)
            self.assertEqual(extended[-1]['reference_index'],5)
            self.assertAlmostEqual(extended[-1]['load'],.0005)
            self.assertEqual(hashlib.sha256((root/'pinn/state_0002/state.zip').read_bytes()).hexdigest(),digest)


if __name__=='__main__':
    unittest.main()
