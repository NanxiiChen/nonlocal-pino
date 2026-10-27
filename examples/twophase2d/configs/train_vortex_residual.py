"""Overshoot-compatible inverse-Jacobian residual experiment; unchanged CH physics."""
from ml_collections import ConfigDict
from examples.twophase2d.configs.train_vortex import get_config as vortex_config


def get_config():
    c = vortex_config()
    c.residual = ConfigDict(dict(mode='both', eta=0.1))
    return c
