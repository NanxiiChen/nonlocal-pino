"""Unchanged physics with exact initial field and bounded exponential lifting."""
from ml_collections import ConfigDict
from examples.particle_corrosion2d.configs.train_physical_slow import get_config as slow_config


def get_config():
    c = slow_config()
    c.phase = ConfigDict(dict(initial_lifting=dict(kind='bounded_exponential', time_scale=.001)))
    c.loss.gradnorm_bias = (1.,)*4
    return c
