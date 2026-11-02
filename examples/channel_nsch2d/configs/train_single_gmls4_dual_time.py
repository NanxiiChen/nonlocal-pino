"""Cold dual-time representation with the independently checked degree4 operator."""
from ml_collections import ConfigDict
from examples.channel_nsch2d.configs.train_single_gmls4 import get_config as baseline


def get_config():
    c=baseline();del c.train.continuation;c.train.iterations=10000
    c.time_embedding=ConfigDict(dict(kind='linear_log',scale=c.lifting.time_scale))
    return c
