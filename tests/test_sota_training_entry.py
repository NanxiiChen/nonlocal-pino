"""Canonical CLI dispatch must preserve overrides and avoid duplicate flags."""
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

from examples.corrosion2d.configs.train_base import get_config
from examples.corrosion2d.training_entry import train_matched, validate_matched_options
from nonlocal_pinn.training.runtime import with_run_seed, configure_gpu_policy


ROOT = Path(__file__).resolve().parents[1]


class CanonicalTrainingEntryTests(unittest.TestCase):
    def test_shared_policy_rejects_unknown_values_before_backend_work(self):
        with patch.dict(os.environ, {'XLA_FLAGS': '--xla_gpu_autotune_level=0'}):
            self.assertEqual(configure_gpu_policy('ordinary'), '--xla_gpu_autotune_level=0')
            self.assertEqual(configure_gpu_policy('deterministic'),
                             '--xla_gpu_autotune_level=0 --xla_gpu_deterministic_ops=true')
            with self.assertRaisesRegex(ValueError, 'Ordinary GPU policy'):
                configure_gpu_policy('ordinary')
        with patch.dict(os.environ, {'XLA_FLAGS': '--xla_gpu_deterministic_ops'}):
            with self.assertRaisesRegex(ValueError, 'Ordinary GPU policy'):
                configure_gpu_policy('ordinary')
            self.assertEqual(configure_gpu_policy('deterministic'),
                             '--xla_gpu_deterministic_ops')
        with self.assertRaisesRegex(ValueError, 'Unknown train.gpu_policy'):
            configure_gpu_policy('automatic')

    def test_ordinary_policy_is_explicit_and_does_not_override_environment(self):
        c=get_config();c.train.gpu_policy='ordinary'
        with patch.dict(os.environ,{'XLA_FLAGS':''}):
            with patch('examples.corrosion2d.residual_experiment.train') as train:
                train_matched(c,'corrosion2d')
                train.assert_called_once_with(c)
                self.assertEqual(os.environ['XLA_FLAGS'],'')
        with patch.dict(os.environ,{'XLA_FLAGS':'--xla_gpu_deterministic_ops=true'}):
            with self.assertRaisesRegex(ValueError,'Ordinary GPU policy'):
                train_matched(c,'corrosion2d')

    def test_seed_is_run_provenance_and_legacy_configs_remain_loadable(self):
        config = get_config()
        self.assertNotIn('seed', config.train)
        resolved = with_run_seed(config, None)
        self.assertEqual(resolved.train.seed, 42)
        self.assertNotIn('seed', config.train)
        self.assertEqual(with_run_seed(config, 2026).train.seed, 2026)
        resolved.train.seed = 7  # old serialized run config
        self.assertEqual(with_run_seed(resolved, None).train.seed, 7)
        self.assertEqual(with_run_seed(resolved, 2026).train.seed, 2026)
        self.assertEqual(resolved.train.seed, 7)
        config.lock()
        self.assertEqual(with_run_seed(config, 2026).train.seed, 2026)
        self.assertNotIn('seed', config.train)
        self.assertTrue(config.is_locked)
        for invalid in (True, -1, 2**32, .5, '2026'):
            with self.assertRaisesRegex(ValueError, 'Run seed'):
                with_run_seed(config, invalid)

    def test_unsupported_legacy_options_fail_before_training(self):
        for section, key, value in (('nl', 'method', 'fd'),
                                    ('sample', 'K_core', 0),
                                    ('loss', 'use_causal', True),
                                    ('loss', 'iface_gamma_init', 1.),
                                    ('loss', 'iface_gamma_decay_steps', 10),
                                    ('loss', 'bias_schedule', ((1, (1.,)*5),)),
                                    ('log', 'save_ckpt', False),
                                    ('log', 'run_diag', True)):
            with self.subTest(section=section, key=key):
                config = get_config()
                config[section][key] = value
                with self.assertRaisesRegex(ValueError, 'Matched training'):
                    validate_matched_options(config)

    def test_direct_and_module_entries_dispatch_once_with_config_overrides(self):
        for case in ('corrosion2d', 'electropolish2d'):
            for module in (False, True):
                with self.subTest(case=case, module=module):
                    target = f'examples/{case}/train.py'
                    entry = (f"runpy.run_module('examples.{case}.train', run_name='__main__')"
                             if module else f"runpy.run_path({target!r}, run_name='__main__')")
                    # Mock only the costly numerical engine, exercising real absl parsing.
                    code = f"""
import os, runpy, sys
from unittest.mock import patch
sys.argv = [{target!r}, '--seed=2026', '--config.train.lr_init=0.0002', '--config.log.run_name=cli-test']
with patch('examples.corrosion2d.residual_experiment.train', return_value=None) as train:
    try:
        {entry}
    except SystemExit as error:
        assert error.code in (None, 0), error.code
    assert train.call_count == 1, train.call_count
    config = train.call_args.args[0]
    assert config.experiment.case == {case!r}
    assert config.train.seed == 2026
    assert config.train.lr_init == 0.0002
    assert config.log.run_name == 'cli-test'
    assert config.train.gpu_policy == 'ordinary'
    assert config.train.compute_dtype == ('float64' if {case!r} == 'corrosion2d' else 'float32')
    assert os.environ['XLA_FLAGS'] == ''
"""
                    env = dict(os.environ, JAX_PLATFORMS='cpu', XLA_FLAGS='')
                    result = subprocess.run([sys.executable, '-c', code], cwd=ROOT,
                                            env=env, capture_output=True, text=True, timeout=60)
                    self.assertEqual(result.returncode, 0, result.stdout+result.stderr)

    def test_melting_cli_resolves_run_seed_before_model_or_filesystem_work(self):
        code = """
import runpy, sys
from pathlib import Path
from unittest.mock import patch
from nonlocal_pinn.training.runtime import with_run_seed, configure_gpu_policy
from examples.icemelting3d.configs.train_base import get_config
assert 'seed' not in get_config().train
sys.argv = ['examples/icemelting3d/train.py', '--seed=2026']
records = []
def resolve(config, seed):
    result = with_run_seed(config, seed)
    records.append(result.train.seed)
    return result
class StopBeforeWrite(Exception): pass
original_mkdir = Path.mkdir
def stop_run_directory(path, *args, **kwargs):
    if str(path).startswith('/root/autodl-tmp/tf-logs/icemelting3d/'):
        raise StopBeforeWrite
    return original_mkdir(path, *args, **kwargs)
with patch('nonlocal_pinn.training.runtime.with_run_seed', side_effect=resolve), \\
     patch('nonlocal_pinn.training.runtime.configure_gpu_policy', wraps=configure_gpu_policy) as gpu, \\
     patch('pathlib.Path.mkdir', new=stop_run_directory):
    try:
        runpy.run_path('examples/icemelting3d/train.py', run_name='__main__')
    except StopBeforeWrite:
        pass
assert records == [2026], records
gpu.assert_called_once_with('ordinary')
"""
        result = subprocess.run([sys.executable, '-c', code], cwd=ROOT,
                                env=dict(os.environ, JAX_PLATFORMS='cpu'),
                                capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)

    def test_adapter_preserves_unrelated_flags_and_rejects_conflicts_and_wrong_case(self):
        c = get_config()
        c.train.gpu_policy = 'deterministic'
        with patch.dict(os.environ, {'XLA_FLAGS': '--xla_gpu_autotune_level=0'}):
            with patch('examples.corrosion2d.residual_experiment.train') as train:
                train_matched(c, 'corrosion2d')
                train.assert_called_once_with(c)
                self.assertEqual(os.environ['XLA_FLAGS'],
                    '--xla_gpu_autotune_level=0 --xla_gpu_deterministic_ops=true')
                train_matched(c, 'corrosion2d')
                self.assertEqual(os.environ['XLA_FLAGS'].count('--xla_gpu_deterministic_ops='), 1)
        with patch.dict(os.environ, {'XLA_FLAGS': '--xla_gpu_deterministic_ops=false'}):
            with patch('examples.corrosion2d.residual_experiment.train') as train:
                with self.assertRaisesRegex(ValueError, 'conflicting'):
                    train_matched(c, 'corrosion2d')
                train.assert_not_called()
        with self.assertRaisesRegex(ValueError, 'received config'):
            train_matched(c, 'electropolish2d')


if __name__ == '__main__':
    unittest.main()
