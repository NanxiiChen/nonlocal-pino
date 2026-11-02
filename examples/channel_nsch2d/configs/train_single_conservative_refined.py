"""Explicit full-state192x80 refinement of the completed conservative pilot."""
from ml_collections import ConfigDict
from examples.channel_nsch2d.configs.train_single_conservative import get_config as baseline


def get_config():
    c=baseline();c.sample.Nx=192;c.sample.Ny=80;c.train.iterations=10000
    c.train.continuation=ConfigDict(dict(kind='refine',
        parent_run='/root/autodl-tmp/tf-logs/channel_nsch2d/20261101-5090-transport-channel-single-sbp-01'))
    return c
