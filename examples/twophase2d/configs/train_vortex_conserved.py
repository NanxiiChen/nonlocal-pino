"""Cubic coordinate with consistent nodal advection and exact grid mass."""
from ml_collections import ConfigDict
from examples.twophase2d.configs.train_vortex_consistent import get_config as consistent_config


def get_config():
    c = consistent_config()
    c.phase = ConfigDict(dict(map='inverse_cubic'))
    return c
