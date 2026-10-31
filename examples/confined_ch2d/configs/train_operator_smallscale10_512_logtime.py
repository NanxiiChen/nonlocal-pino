"""Stable-part metric with the existing generic logarithmic IC/time coordinate."""
from ml_collections import ConfigDict
from examples.confined_ch2d.configs.train_operator_smallscale10_512_stable import get_config as baseline


def get_config():
    c=baseline()
    c.phase=ConfigDict(dict(lifting='logarithmic_additive',initial_time_scale=.05))
    return c
