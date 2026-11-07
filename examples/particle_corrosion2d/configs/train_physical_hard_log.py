"""Exact IC and resolved initial transient within one global-time network."""
from ml_collections import ConfigDict
from examples.particle_corrosion2d.configs.train_physical_hard import get_config as hard_config


def get_config():
    c = hard_config()
    c.sample.temporal = ConfigDict(dict(kind='logarithmic_stratified', time_scale=.01))
    c.arch.time_coordinate = ConfigDict(dict(kind='logarithmic', time_scale=.01))
    return c
