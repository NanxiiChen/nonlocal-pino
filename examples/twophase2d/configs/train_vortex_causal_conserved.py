"""Causal metric with consistent nodal advection and exact grid mass."""
from ml_collections import ConfigDict
from examples.twophase2d.configs.train_vortex_conserved import get_config as conserved_config


def get_config():
    c = conserved_config()
    c.preconditioner = ConfigDict(dict(kind='ch_causal',
        time_scale=c.physical.terminal_time, curvature=2.))
    return c
