"""Frozen bounded droplet metric on the unchanged audited15k coupled parent."""
from ml_collections import ConfigDict
from examples.channel_nsch2d.configs.train_single_lbfgs import get_config as baseline


def get_config():
    c=baseline()
    c.droplet_metric=ConfigDict(dict(kind='frozen_soft_phase',contrast=24.,temperature=.2,
        fields=('phase','chemical','momentum_x','momentum_y'),
        normalization='interior_mean',source='parent_model'))
    return c
