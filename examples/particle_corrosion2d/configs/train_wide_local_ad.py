"""Matched local-AD control for the width192 / 20000-update experiment."""
from examples.particle_corrosion2d.configs.train_wide import get_config as wide_config


def get_config():
    c=wide_config()
    c.nl.method='local_ad'
    c.loss.gradnorm_bias=tuple(c.loss.gradnorm_bias)+(1.,1.)
    return c
