"""Conserved cubic original residual; include t=0 as one ordinary PDE time."""
from examples.twophase2d.configs.train_vortex_conserved import get_config as conserved_config


def get_config():
    c = conserved_config()
    c.sample.include_initial_time = True
    return c
