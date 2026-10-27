"""Convexified-chemical coordinate plus frozen mixed-CH residual preconditioner."""
from ml_collections import ConfigDict
from examples.twophase2d.configs.train_vortex_cubic import get_config as cubic_config


def get_config():
    c = cubic_config()
    c.preconditioner = ConfigDict(dict(kind='ch_schur', time_scale=0.1, curvature=2.))
    return c
