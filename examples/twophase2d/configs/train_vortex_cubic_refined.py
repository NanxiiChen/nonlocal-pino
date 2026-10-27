"""Convexified-chemical phase coordinate with PDE-only refinement."""
from ml_collections import ConfigDict
from examples.twophase2d.configs.train_vortex_refined import get_config as refined_config


def get_config():
    c = refined_config()
    c.phase = ConfigDict(dict(map='inverse_cubic'))
    return c
