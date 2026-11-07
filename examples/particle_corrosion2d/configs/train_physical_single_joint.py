"""Matched joint-PDE control without temporal causal discounts."""
from examples.particle_corrosion2d.configs.train_physical_single_gn_l2 import get_config as norm_config


def get_config():
    c=norm_config()
    c.loss.use_causal=False
    del c.causal
    return c
