"""Strict complete-state continuation of the width96 capacity control to20k."""
from ml_collections import ConfigDict
from examples.confined_ch2d.configs.train_operator_smallscale10_512_batch_wide import get_config as baseline


def get_config():
    c=baseline()
    c.train.iterations=20000
    c.train.continuation=ConfigDict(dict(parent_run=
        '/root/autodl-tmp/tf-logs/confined_ch2d/20261030-confined-geometry-b16-t16-wide96-soap-01'))
    return c
