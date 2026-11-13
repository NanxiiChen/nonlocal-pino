"""The user-facing entrypoint must retain the accepted physical/nonlocal recipe."""
from pathlib import Path
import unittest
import os
from unittest.mock import patch

from examples.particle_corrosion2d.configs.train_default import get_config
from examples.particle_corrosion2d.configs.train_physical_spaced_w256_fixed_adjoint import get_config as accepted
from examples.particle_corrosion2d.problem import validate
from examples.particle_corrosion2d import train


class ParticleDefaultTests(unittest.TestCase):
    def test_explicit_execution_control_preserves_other_flags_and_rejects_conflict(self):
        from examples.particle_corrosion2d.execution import configure_execution
        from examples.particle_corrosion2d.configs.train_physical_spaced_w256_fixed_adjoint import get_config as fixed
        c=fixed()
        with patch.dict(os.environ,{'XLA_FLAGS':'--xla_gpu_enable_triton_gemm=true'}):
            configure_execution(c);configure_execution(c)
            self.assertEqual(os.environ['XLA_FLAGS'],
                '--xla_gpu_enable_triton_gemm=true --xla_gpu_autotune_level=0')
        with patch.dict(os.environ,{'XLA_FLAGS':'--xla_gpu_autotune_level=4'}):
            with self.assertRaisesRegex(ValueError,'conflicts'):
                configure_execution(c)

    def test_default_has_selected_physics_geometry_and_derivative(self):
        c=get_config();validate(c)
        self.assertEqual(c.to_dict(),accepted().to_dict())
        self.assertEqual(c.nl.method,'nonlocal')
        self.assertEqual(c.particles.count,6)
        self.assertEqual(c.arch.width,256)
        self.assertEqual(c.nondim.T_c,350.)
        self.assertEqual(c.physical.L,2e-9)
        self.assertEqual(c.train.iterations,20000)
        self.assertIn('online',c.operator)
        self.assertNotIn('seed',c.train)

    def test_train_cli_without_config_selects_the_accepted_preset(self):
        arguments=['train','--inputs','/tmp/inputs','--reference','/tmp/reference','--output','/tmp/run']
        with patch('sys.argv',arguments),patch.object(train,'train') as execute:
            train.main()
        execute.assert_called_once()
        args=execute.call_args.args
        self.assertEqual(args[0].to_dict(),accepted().to_dict())
        self.assertEqual(args[4],Path(train.__file__).resolve().parent/'configs/train_default.py')
        self.assertIsNone(args[10])


if __name__=='__main__':unittest.main()
