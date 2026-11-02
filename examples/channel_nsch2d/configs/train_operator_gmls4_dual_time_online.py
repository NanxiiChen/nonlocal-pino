"""Untrained online extension of the cold degree4/dual-time control."""
from ml_collections import ConfigDict
from examples.channel_nsch2d.configs.train_operator_gmls4_online import get_config as baseline


def get_config():
    c=baseline();c.time_embedding=ConfigDict(dict(kind='linear_log',scale=c.lifting.time_scale))
    c.operator.validation_manifest='/root/autodl-tmp/data/channel_nsch2d/unseen-geometry16-offgrid-20261101/manifest.json'
    return c
