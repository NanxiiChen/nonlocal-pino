"""Twice the evolution rate on [0, 1], with unchanged kinetic ratios/width."""
from examples.particle_corrosion2d.configs.train_online import get_config as online_config


def get_config():
    c = online_config()
    c.physical.n_ac1 *= 2.0
    c.physical.n_ac2 *= 2.0
    c.physical.n_ac3 *= 2.0
    c.physical.n_ch *= 2.0
    return c
