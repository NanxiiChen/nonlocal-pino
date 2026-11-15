"""Common local-reference identity, metric order and matched temporal measure."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from examples.confined_ch2d.configs.train_operator_compare_nonlocal import get_config
from examples.confined_ch2d.operator_comparison import validate_comparison
from examples.confined_ch2d.operator_comparison_monitor import load,metrics
from examples.confined_ch2d.operator_problem import specification
from docs.tasks.spinodal_local_reference import relative


class ComparisonMonitorTests(unittest.TestCase):
    def test_channel_norm_then_mean_and_finite_rejection(self):
        ref=dict(phi=np.arange(1.,13.).reshape(3,4),mu=np.arange(13.,25.).reshape(3,4))
        target=np.stack((ref['phi'],ref['mu']),axis=-1)
        pred=target*np.array([1.1,1.3]);result=metrics(pred,ref)
        self.assertAlmostEqual(result['phi'],.1,places=13)
        self.assertAlmostEqual(result['chemical'],.3,places=13)
        self.assertAlmostEqual(result['channel_mean'],.2,places=13)
        self.assertAlmostEqual(result['chemical_positive_times'],.3,places=13)
        with self.assertRaises(ValueError):metrics(pred[:-1],ref)
        pred[0,0,0]=np.nan
        with self.assertRaises(ValueError):metrics(pred,ref)
        with self.assertRaises(ValueError):relative(pred[...,0],ref['phi'],np.ones(4))

    def test_reference_identity_and_failed_refinement_are_not_ignored(self):
        c=get_config();bank=dict(test_indices=[1]);refs={1:dict(points=np.array([[.2,.4],[.6,.7]]),
            volumes=np.array([.1,.2]),time=np.array([0.,1.,3.]),holes=np.array([[.4,.5,.07],[.6,.5,.08]]),
            phi=np.ones((3,2)),mu=np.ones((3,2))*2)}
        with tempfile.TemporaryDirectory() as directory:
            c.operator.comparison_reference.path=directory;root=Path(directory)
            np.savez(root/'case.npz',**refs[1])
            manifest=dict(status='passed',kind='independent_mixed_p2_bdf2',specification=specification(c),
                indices=[1],cases=[dict(index=1,status='passed',file='case.npz')])
            path=root/'manifest.json';path.write_text(json.dumps(manifest))
            _,loaded=load(c,bank,refs)
            np.testing.assert_array_equal(loaded[1]['phi'],refs[1]['phi'])
            manifest['cases'][0]['status']='failed';path.write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):load(c,bank,refs)
            manifest['cases'][0].update(status='passed',file='../case.npz');path.write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):load(c,bank,refs)
            manifest['cases'][0]['file']='case.npz';manifest['specification']['physical']['epsilon']*=2
            path.write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):load(c,bank,refs)

    def test_default_unchanged_and_all_controls_match_time_measure(self):
        from examples.confined_ch2d.configs.train_operator import get_config as default
        from examples.confined_ch2d.configs.train_operator_compare_local_ad import get_config as ad
        from examples.confined_ch2d.configs.train_operator_compare_fem import get_config as fem
        self.assertEqual(default().loss.initial_residual_fraction,.2)
        for make in (get_config,ad,fem):
            c=make();validate_comparison(c)
            self.assertEqual(c.loss.initial_residual_fraction,0.)
            self.assertEqual(c.sample.Nt,16)
            self.assertEqual(c.train.iterations,50000)
            c.loss.initial_residual_fraction=.2
            with self.assertRaises(ValueError):validate_comparison(c)


if __name__=='__main__':unittest.main()
