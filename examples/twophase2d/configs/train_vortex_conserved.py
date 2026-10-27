"""Direct phase output with conservative advection and exact grid mass.

Opt in to the cubic coordinate with --config.phase.map=inverse_cubic.
"""
from examples.twophase2d.configs.train_vortex_consistent import get_config as consistent_config


def get_config():
    c = consistent_config()
    return c
