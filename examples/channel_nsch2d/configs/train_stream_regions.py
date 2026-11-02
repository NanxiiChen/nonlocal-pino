"""Positive droplet/carrier/interface-balanced phase and chemical PDE metric."""
from ml_collections import ConfigDict
from examples.channel_nsch2d.configs.train_operator import get_config as baseline


def get_config():
    c=baseline()
    c.phase_metric=ConfigDict(dict(kind='balanced_regions',region_fraction=.5,
        minimum_fraction=.001,stop_weights=True,fields=('phase','chemical')))
    return c
