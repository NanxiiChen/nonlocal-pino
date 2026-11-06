"""Capacity/optimization control after similar training and unseen errors."""
from examples.particle_corrosion2d.configs.train_base import get_config as base_config


def get_config():
    c=base_config()
    c.arch.width=192
    c.arch.spatial_features=96
    c.train.iterations=20000
    c.train.lr_decay_every=3000
    c.log.eval_every=1000
    c.log.checkpoint_every=5000
    return c
