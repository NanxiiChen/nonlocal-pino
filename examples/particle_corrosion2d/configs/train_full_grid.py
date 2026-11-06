"""Whole-mesh PDE quadrature with the unchanged rolling-geometry recipe."""
from ml_collections import ConfigDict

from examples.particle_corrosion2d.configs.train_online import get_config as online_config


def get_config():
    c = online_config()
    c.sample.full_grid = ConfigDict(dict(nodes=4608, neighbors=64))
    return c
