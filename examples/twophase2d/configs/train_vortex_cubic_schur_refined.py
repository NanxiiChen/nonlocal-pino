"""Cubic phase coordinate and CH Schur metric with PDE-only refinement."""
from ml_collections import ConfigDict
from examples.twophase2d.configs.train_vortex_cubic_refined import get_config as refined_config


def get_config():
    c = refined_config()
    c.preconditioner = ConfigDict(dict(kind='ch_schur', time_scale=0.1, curvature=2.))
    return c
