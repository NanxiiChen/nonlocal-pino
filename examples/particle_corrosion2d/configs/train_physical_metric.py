"""Slower Adam and unscaled physical concentration data errors."""
from examples.particle_corrosion2d.configs.train_physical_slow import get_config as slow_config


def get_config():
    c = slow_config()
    c.loss.concentration_scale = 1.0
    return c
