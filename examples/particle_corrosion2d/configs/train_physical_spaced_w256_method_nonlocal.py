"""Paired rolling derivative study; selected physics, with sparse checkpoints."""
from examples.particle_corrosion2d.configs.train_physical_spaced_w256 import (
    get_config as selected_config,
)


def get_config():
    c = selected_config()
    c.operator.method_comparison = 'rolling_spatial_v1'
    c.log.checkpoint_every = 5000
    return c
