"""Exact fixed-matrix parameter adjoints; original forward nonlocal differences."""
from examples.particle_corrosion2d.configs.train_physical_spaced_w256 import get_config as baseline


def get_config():
    c=baseline()
    c.nl.adjoint='dense_fixed'
    c.train.autotune_level=0
    return c
