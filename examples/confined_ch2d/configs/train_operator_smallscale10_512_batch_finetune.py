"""Explicit .3 learning-rate scaling after the complete20k B16/Nt16 parent."""
from ml_collections import ConfigDict
from examples.confined_ch2d.configs.train_operator_smallscale10_512_batch import get_config as baseline


def get_config():
    c=baseline()
    c.train.iterations=25000
    c.train.lr_init*=.3
    c.train.lr_min*=.3
    c.train.continuation=ConfigDict(dict(
        parent_run='/root/autodl-tmp/tf-logs/confined_ch2d/20261030-confined-geometry-b16-t16-soap-continue-01',
        learning_rate_factor=.3))
    return c
