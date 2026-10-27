"""Invertible convexified-chemical phase coordinate, with original residuals."""
from ml_collections import ConfigDict
from examples.twophase2d.configs.train_vortex import get_config as vortex_config


def get_config():
    c = vortex_config()
    c.phase = ConfigDict(dict(map='inverse_cubic'))
    return c
