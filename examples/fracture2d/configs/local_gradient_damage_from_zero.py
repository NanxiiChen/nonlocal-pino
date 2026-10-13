"""Matched local-gradient/lumped-P1 damage control; own zero history."""
from ml_collections import ConfigDict
from examples.fracture2d.configs.local_damage_trial_from_zero import get_config as base


def get_config():
    c = base()
    c.comparison = ConfigDict()
    c.comparison.loads = [float(i * c.data.load_step) for i in range(1, 76)]
    c.comparison.output_dir = '/root/autodl-tmp/tf-logs/fracture2d/local_gradient_from_zero_20261011'
    c.comparison.log_every = 16
    return c
