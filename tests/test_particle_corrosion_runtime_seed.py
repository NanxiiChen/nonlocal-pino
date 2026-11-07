"""Replay is runtime provenance; default runs remain fresh and controls win."""
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np

from examples.particle_corrosion2d.train import resolve_runtime_seed,train
from examples.particle_corrosion2d.configs.train_physical_spaced import get_config


class RuntimeSeedTests(unittest.TestCase):
    def test_absent_seed_stays_random_and_explicit_seed_is_not_config(self):
        with patch('examples.particle_corrosion2d.train.secrets.randbits',side_effect=[17,91]) as random:
            self.assertEqual(resolve_runtime_seed(None),17)
            self.assertEqual(resolve_runtime_seed(None),91)
            self.assertEqual(resolve_runtime_seed(np.uint32(42)),42)
            self.assertEqual(random.call_count,2)
        self.assertNotIn('seed',get_config().train)

    def test_invalid_runtime_seeds_fail(self):
        for value in [True,-1,2**32,.5,'42']:
            with self.assertRaisesRegex(ValueError,'Runtime seed'):
                resolve_runtime_seed(value)

    def test_explicit_seed_cannot_override_comparison_before_backend_setup(self):
        c=get_config()
        with patch('examples.particle_corrosion2d.train.configure_gpu_policy') as configure:
            with self.assertRaisesRegex(ValueError,'cannot override'):
                train(c,Path('/tmp/unused'),Path('/tmp/unused'),Path('/tmp/unused'),
                      Path('/tmp/unused'),Path('/tmp/control'),runtime_seed=17)
            configure.assert_not_called()


if __name__=='__main__':unittest.main()
