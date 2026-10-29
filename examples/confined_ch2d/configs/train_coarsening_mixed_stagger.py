"""One global shared-head network; alternate PDE losses, then train jointly."""
from ml_collections import ConfigDict
from examples.confined_ch2d.configs.train_coarsening_mixed_joint import get_config as baseline


def get_config():
    c=baseline()
    c.train.loss_balance=ConfigDict(dict(kind='stagger',switch_every=25,
        stagger_steps=4000,refinement='joint'))
    return c
