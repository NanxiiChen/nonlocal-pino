"""Matched continuous online width128counterpart with linear/log time features."""
from ml_collections import ConfigDict
from examples.channel_nsch2d.configs.train_operator_width128_online import get_config as baseline


def get_config():
    c=baseline();c.time_embedding=ConfigDict(dict(kind='linear_log',scale=c.lifting.time_scale))
    return c
