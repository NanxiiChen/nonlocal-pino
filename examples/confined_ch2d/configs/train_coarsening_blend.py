"""General decaying IC lift; physical case and historical default unchanged."""
from ml_collections import ConfigDict
from examples.confined_ch2d.configs.train_coarsening import get_config as baseline


def get_config():
    c=baseline()
    c.phase=ConfigDict(dict(lifting='exponential_blend',initial_time_scale=.15))
    return c
