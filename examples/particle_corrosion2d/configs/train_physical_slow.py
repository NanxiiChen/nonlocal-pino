"""Unchanged SI physics, slower electro-style Adam schedule, short screening."""
from examples.particle_corrosion2d.configs.train_physical import get_config as physical_config


def get_config():
    c = physical_config()
    c.train.iterations = 5000
    c.train.lr_decay_every = 1000
    c.train.lr_decay_rate = .95
    return c
