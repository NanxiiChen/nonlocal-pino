"""Unequal interacting coherent particles; explicit harder periodic variant."""
from ml_collections import ConfigDict
from examples.precipitate2d.configs.train_base import get_config as base_config


def get_config():
    c = base_config()
    c.initial = ConfigDict(dict(kind='multiple',
        centers=((.35, .32), (.67, .36), (.40, .68)),
        levels=(.18, .11, .065), width=.035, background=-.95))
    c.physical.lame_lambda = 9800.
    c.physical.shear_modulus = 4900.
    c.physical.terminal_time = 4.
    c.reference.local_grid = 191
    c.reference.path = '/root/autodl-tmp/data/precipitate2d/results_periodic_multiple.npz'
    return c
