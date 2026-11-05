"""A larger network and longer optimization for the same stronger material prior."""
from examples.heterogeneous_corrosion2d.configs.train_stronger import get_config as stronger_config


def get_config():
    c = stronger_config()
    c.arch.width = 96
    c.arch.spatial_features = 48
    c.train.iterations = 10000
    c.train.lr_decay_every = 2000
    return c
