"""Online counterpart with complete old linear initialization and new log capacity."""
from ml_collections import ConfigDict
from examples.channel_nsch2d.configs.train_operator_time_refined_online import get_config as baseline


def get_config():
    c=baseline();c.arch.time_features=16
    c.time_embedding=ConfigDict(dict(kind='linear_log',scale=c.lifting.time_scale,initialization='preserve_linear'))
    return c
