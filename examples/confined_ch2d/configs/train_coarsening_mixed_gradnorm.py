"""Shared-head global CH with the project's inverse-L1-gradient balancing."""
from ml_collections import ConfigDict
from examples.confined_ch2d.configs.train_coarsening_mixed_joint import get_config as baseline


def get_config():
    c=baseline()
    c.train.loss_balance=ConfigDict(dict(kind='gradient_norm',update_every=25,
        alpha=1.,norm_floor=1e-6,norm_ceiling=1e6,weight_floor=1e-6,
        weight_cap=10000.,refinement='freeze'))
    return c
