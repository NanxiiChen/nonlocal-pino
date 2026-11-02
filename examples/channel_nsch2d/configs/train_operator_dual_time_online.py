"""Untrained online extension of the explicit dual-time single-geometry control."""
from ml_collections import ConfigDict
from examples.channel_nsch2d.configs.train_operator_conservative_online import get_config as baseline


def get_config():
    c=baseline();c.time_embedding=ConfigDict(dict(kind='linear_log',scale=c.lifting.time_scale))
    return c
