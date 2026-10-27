"""Causal frozen mixed-CH block inverse, with the resolved cubic coordinate."""
from ml_collections import ConfigDict
from examples.twophase2d.configs.train_vortex_cubic_resolved import get_config as resolved_config


def get_config():
    c = resolved_config()
    c.preconditioner = ConfigDict(dict(kind='ch_causal',
        time_scale=c.physical.terminal_time, curvature=2.))
    return c
