"""Explicit paired execution settings; the production preset remains unchanged."""
import os
import shlex


def validate_comparison(c):
    if (c.operator.action != 'train' or 'continuation' in c.train
            or c.train.gpu_policy != 'ordinary' or 'autotune_level' not in c.train
            or type(c.train.autotune_level) is not int or c.train.autotune_level != 0):
        raise ValueError('Three-method control requires cold train, ordinary policy and explicit autotune_level=0')
    if c.loss.initial_residual_fraction != 0.:
        raise ValueError('Matched comparisons impose the PDE only at positive times; hard IC stays exact')
    if c.operator.spatial_discretization == 'local_ad':
        from examples.confined_ch2d.operator_local_ad import validate_local_ad
        validate_local_ad(c)
    elif c.operator.spatial_discretization == 'fem_p1_lumped':
        from examples.confined_ch2d.operator_fem_assembly import validate_fem
        validate_fem(c)
    elif c.operator.spatial_discretization != 'nonlocal':
        raise ValueError('Unknown Spinodal comparison method')


def configure_comparison(c):
    validate_comparison(c)
    flags = os.environ['XLA_FLAGS'] if 'XLA_FLAGS' in os.environ else ''
    found = [arg for arg in shlex.split(flags) if arg.startswith('--xla_gpu_autotune_level=')]
    if any(arg != '--xla_gpu_autotune_level=0' for arg in found):
        raise ValueError('Inherited autotuning conflicts with the explicit comparison setting')
    if not found:
        os.environ['XLA_FLAGS'] = (flags+' --xla_gpu_autotune_level=0').strip()
