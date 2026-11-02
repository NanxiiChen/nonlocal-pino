"""Matched cold width128control: only the existing linear/log time coordinate."""
from ml_collections import ConfigDict
from examples.channel_nsch2d.configs.train_single_width128 import get_config as baseline


def get_config():
    c=baseline();c.time_embedding=ConfigDict(dict(kind='linear_log',scale=c.lifting.time_scale))
    return c
