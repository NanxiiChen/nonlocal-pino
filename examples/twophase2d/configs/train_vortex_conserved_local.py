"""Matched local AD diffusion control for the recommended conserved vortex."""
from examples.twophase2d.configs.train_vortex_conserved import get_config as conserved_config


def get_config():
    c = conserved_config()
    c.nl.method = 'local_ad'
    return c
