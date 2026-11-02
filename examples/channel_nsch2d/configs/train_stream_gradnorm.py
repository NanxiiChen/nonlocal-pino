"""Matched streamfunction control with explicit repo gradient-norm balancing."""
from ml_collections import ConfigDict
from examples.channel_nsch2d.configs.train_operator import get_config as baseline


def get_config():
    c=baseline()
    c.balance=ConfigDict(dict(kind='gradnorm',norm='l1',epsilon=1e-6,cap=1000.,
        new_weight_fraction=.9,update_every=100))
    return c
