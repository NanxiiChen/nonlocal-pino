"""Monotone log-time coordinate and hard-IC lift; no trajectory prior."""
from ml_collections import ConfigDict
from examples.confined_ch2d.configs.train_coarsening_causal import get_config as causal


def get_config():
    c=causal()
    c.phase=ConfigDict(dict(lifting='logarithmic_additive',initial_time_scale=.005))
    return c
