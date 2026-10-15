"""Isolate positive baseline reparameterization, with identical feature centers."""
from examples.fracture2d._legacy.configs.compact_damage_multiscale import get_config as base


def get_config():
    c = base()
    c.compact.output_dir = '/root/autodl-tmp/tf-logs/fracture2d/compact_damage_positive_fixedH_20261011'
    c.compact.baseline_mode = 'positive_conditioned'
    return c
