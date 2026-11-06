"""Matched local-AD control of the current random-particle operator preset."""
from examples.particle_corrosion2d.configs.train_base import get_config as base_config


def get_config():
    c = base_config()
    c.nl.method = 'local_ad'
    c.loss.gradnorm_bias = tuple(c.loss.gradnorm_bias)+(1., 1.)
    return c
