"""State-scaled stopping and progress-aware blocks without manual load phases."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from ml_collections import ConfigDict
from examples.fracture2d.learning_rate import BudgetLearningRate
from examples.fracture2d.loading import run, validate
from examples.fracture2d.stagger import damage_change_limit, field_stop
from tests.test_fracture2d_budget_learning_rate import controlled, stop_after_drop


def controller():
    return BudgetLearningRate(ConfigDict(dict(mode='progress_blocks',
        rates=[1e-4,3e-5,1e-5], outers_per_rate=128,
        max_blocks_per_load=4, minimum_reaction_decrease=.01)),
        initial_rate=1e-4, minimum_rate=1e-4, decay_rate=1.)


class ProgressStoppingTests(unittest.TestCase):
    def test_scaled_threshold_keeps_displacement_gate_and_small_damage_floor(self):
        a = ConfigDict(dict(damage_tolerance=1e-5,
            damage_stop=dict(mode='scaled_max', offset=.01, factor=.001)))
        self.assertAlmostEqual(damage_change_limit(a,damage_max=0.),1e-5)
        self.assertAlmostEqual(damage_change_limit(a,damage_max=1.),.00101)
        change = dict(damage_rms=5e-4, displacement_rms_over_load=5e-4)
        kwargs = dict(displacement_limit=1e-4, damage_limit=damage_change_limit(a,damage_max=1.),
            sweep=10,minimum_sweeps=3,consecutive_passes=3,previous_consecutive=2)
        self.assertFalse(field_stop(change,**kwargs)['converged'])
        change['displacement_rms_over_load'] = 1e-5
        self.assertTrue(field_stop(change,**kwargs)['converged'])
        with self.assertRaisesRegex(ValueError,'damage maximum'):
            damage_change_limit(a,damage_max=1.01)
        del a.damage_stop
        self.assertEqual(damage_change_limit(a,damage_max=.9),1e-5)

    def test_progress_extends_even_at_minimum_but_caps_budget_and_restores_policy(self):
        p = controller()
        self.assertEqual(p.progress_action(block=1,previous_reaction=630.,reaction=470.)['action'],'continue')
        self.assertEqual(p.index,0)
        self.assertEqual(p.progress_action(block=2,previous_reaction=1.55,reaction=1.56)['action'],'lower')
        saved = p.snapshot()
        q = controller(); q.restore(saved)
        self.assertEqual(q.snapshot(),saved)
        q.lower()
        self.assertEqual(q.progress_action(block=2,previous_reaction=330.,reaction=210.)['action'],'continue')
        self.assertEqual(q.progress_action(block=3,previous_reaction=1.55,reaction=1.56)['action'],'stop_minimum')
        self.assertEqual(q.progress_action(block=4,previous_reaction=330.,reaction=210.)['action'],'stop_budget')
        saved['max_blocks_per_load'] = 5
        with self.assertRaisesRegex(ValueError,'differs'):
            controller().restore(saved)

    def test_progress_boundary_sync_and_endpoint_resume_preserve_actual_trajectory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); c = controlled(root)
            c.train.rate_control.mode = 'progress_blocks'
            c.train.rate_control.max_blocks_per_load = 4
            c.train.rate_control.minimum_reaction_decrease = .01
            c.loading.damage_stop = ConfigDict(dict(mode='scaled_max',offset=.01,factor=.001))
            validate(c)
            with contextlib.redirect_stdout(io.StringIO()), patch(
                    'examples.fracture2d.loading.field_stop',side_effect=stop_after_drop):
                run(c,damage_solver='pinn')
            base = Path(c.output.root)
            rows = json.loads((base/'result.json').read_text())['curve']
            self.assertEqual([r['sweep'] for r in rows],[3,3])
            self.assertEqual([r['mechanics_calls'] for r in rows],[5,5])
            c.output.root = str(root/'interrupted')
            from examples.fracture2d.storage import atomic_json
            def interrupt(path,data):
                atomic_json(path,data)
                if path.name=='result.json' and 'curve' in data and len(data['curve'])==1:
                    raise InterruptedError('endpoint')
            with contextlib.redirect_stdout(io.StringIO()), patch(
                    'examples.fracture2d.loading.field_stop',side_effect=stop_after_drop), patch(
                    'examples.fracture2d.loading.atomic_json',side_effect=interrupt):
                with self.assertRaises(InterruptedError):run(c,damage_solver='pinn')
            c.output.resume = True
            with contextlib.redirect_stdout(io.StringIO()), patch(
                    'examples.fracture2d.loading.field_stop',side_effect=stop_after_drop):
                run(c,damage_solver='pinn')
            actual = Path(c.output.root)/'state_0002'
            for name in ('network.eqx','optimizer.eqx','rate_control.json'):
                self.assertEqual((base/'state_0002'/name).read_bytes(),(actual/name).read_bytes())
            with np.load(base/'state_0002/fields.npz') as a,np.load(actual/'fields.npz') as b:
                for key in a.files:np.testing.assert_array_equal(a[key],b[key])


if __name__=='__main__':unittest.main()
