"""Matched local spectral diffusion control for the multiple-particle variant."""
from examples.precipitate2d.configs.train_multiple import get_config as multiple_config


def get_config():
    c = multiple_config()
    c.nl.method = 'local_spectral'
    return c
