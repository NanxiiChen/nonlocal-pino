"""Fresh scaled shared-head global control for matched loss-balancing runs."""
from ml_collections import ConfigDict
from examples.confined_ch2d.configs.train_coarsening_global_mixed import get_config as baseline


def get_config():
    c=baseline()
    c.train.loss_balance=ConfigDict(dict(kind='fixed',refinement='joint'))
    return c
