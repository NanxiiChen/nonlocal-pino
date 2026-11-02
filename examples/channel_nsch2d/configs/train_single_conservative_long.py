"""Strict full-state continuation of the192x80 stage, global10000->20000."""
from ml_collections import ConfigDict
from examples.channel_nsch2d.configs.train_single_conservative_refined import get_config as baseline


def get_config():
    c=baseline();c.train.iterations=20000
    c.train.continuation=ConfigDict(dict(kind='strict',
        parent_run='/root/autodl-tmp/tf-logs/channel_nsch2d/20261101-5090-refinement-channel-single-sbp-refined-01'))
    return c
