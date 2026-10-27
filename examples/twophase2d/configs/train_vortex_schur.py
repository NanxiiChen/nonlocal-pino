"""Frozen mixed-CH Schur metric, with unchanged physical phase representation."""
from ml_collections import ConfigDict
from examples.twophase2d.configs.train_vortex import get_config as vortex_config


def get_config():
    c = vortex_config()
    c.preconditioner = ConfigDict(dict(kind='ch_schur', time_scale=0.1, curvature=2.))
    return c
