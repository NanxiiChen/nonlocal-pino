"""Paired 350 s control with half the PDE cores within a horizon of ceramic walls."""
from ml_collections import ConfigDict

from examples.particle_corrosion2d.configs.train_physical_spaced_w256 import get_config as baseline


def get_config():
    c = baseline()
    c.sample.core_wall = ConfigDict(dict(distance=c.nl.delta, count=c.sample.K_core//2))
    return c
