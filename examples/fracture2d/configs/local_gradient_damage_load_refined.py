"""Matched local-gradient control on the existing quarter-step loading grid."""
from ml_collections import ConfigDict
from examples.fracture2d.configs.local_damage_load_refined import get_config as base


def get_config():
    c = base()
    c.comparison = ConfigDict()
    c.comparison.loads = [float(i * c.data.load_step) for i in c.data.reference_indices]
    c.comparison.output_dir = '/root/autodl-tmp/tf-logs/fracture2d/local_gradient_load_refined_20261011'
    c.comparison.log_every = 16
    return c
