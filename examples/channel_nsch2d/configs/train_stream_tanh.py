"""Scaled-tanh output only, preserving the original physical residual metric."""
from ml_collections import ConfigDict
from examples.channel_nsch2d.configs.train_operator import get_config as baseline


def get_config():
    c=baseline()
    c.phase_coordinate=ConfigDict(dict(kind='scaled_tanh',amplitude=1.25))
    return c
