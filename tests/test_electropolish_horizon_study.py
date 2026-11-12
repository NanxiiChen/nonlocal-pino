"""Common targets must remain physical-identical when the training graph changes."""
import unittest
from unittest.mock import patch
import numpy as np
from ml_collections import ConfigDict
from examples.electropolish2d.configs.train_operator import get_config
from examples.electropolish2d.horizon_study import channel_relative_errors, load_monitor, monitor_config


class HorizonStudyTests(unittest.TestCase):
    def config(self):
        c = get_config()
        c.operator.horizon_study = ConfigDict(dict(monitor_Nx=101, monitor_Ny=51, monitor_delta=.02))
        c.sample.Nx, c.sample.Ny, c.nl.delta = 201, 101, .01
        return c

    def test_only_monitor_discretization_changes_and_original_is_untouched(self):
        c = self.config()
        c.lock()
        original = c.to_dict()
        m = monitor_config(c)
        self.assertEqual(c.to_dict(), original)
        self.assertEqual((m.sample.Nx, m.sample.Ny, m.nl.delta), (101, 51, .02))
        for section in ('physical', 'nondim', 'surface', 'arch'):
            self.assertEqual(c[section], m[section])
        self.assertNotIn('horizon_study', m.operator)

    def test_physical_mismatch_still_reaches_strict_dataset_validator(self):
        c = self.config()
        c.physical.alpha_phi *= 2
        def strict_loader(m):
            self.assertEqual(m.physical.alpha_phi, c.physical.alpha_phi)
            raise ValueError('physical/input specification mismatch')
        with patch('examples.electropolish2d.operator_data.load_dataset', side_effect=strict_loader):
            with self.assertRaisesRegex(ValueError, 'specification mismatch'):
                load_monitor(c)

    def test_empty_support_and_unknown_options_rejected(self):
        c = self.config()
        c.nl.delta = .005
        with self.assertRaisesRegex(ValueError, 'spacing'):
            monitor_config(c)
        c = self.config()
        c.operator.horizon_study.silent_override = True
        with self.assertRaisesRegex(ValueError, 'explicit'):
            monitor_config(c)

    def test_metric_normalizes_channels_before_averaging(self):
        r = np.ones((3, 2, 4, 2)); r[..., 1] = 100
        p = r * np.array([1.1, 1.3])
        result = channel_relative_errors(p, r)
        self.assertAlmostEqual(result['channel_mean'], .2)
        scale = np.array([9., .002])
        self.assertAlmostEqual(channel_relative_errors(p*scale, r*scale)['channel_mean'], .2)
        r[1] *= 2
        exact = np.sqrt(np.sum((p-r)**2, axis=(0, 1, 2))/np.sum(r*r, axis=(0, 1, 2))).mean()
        self.assertAlmostEqual(channel_relative_errors(p, r)['channel_mean'], exact)


if __name__ == '__main__':
    unittest.main()
