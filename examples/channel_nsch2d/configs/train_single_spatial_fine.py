"""Spatial-only384x160refinement of the audited dt.025SOAP parent."""
from ml_collections import ConfigDict
from examples.channel_nsch2d.configs.train_single_time_refined import get_config as baseline


def get_config():
    c=baseline();c.sample.Nx=384;c.sample.Ny=160;c.train.iterations=20000
    c.train.continuation=ConfigDict(dict(kind='refine',
        parent_run='/root/autodl-tmp/tf-logs/channel_nsch2d/20261101-5090-time-refine-single-time-01'))
    return c
