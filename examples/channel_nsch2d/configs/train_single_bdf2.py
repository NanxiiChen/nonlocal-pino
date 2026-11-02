"""One fixed geometry, one global-time operator, BE-started BDF2 residuals."""
from ml_collections import ConfigDict
from examples.channel_nsch2d.configs.train_operator import get_config as baseline


def get_config():
    c=baseline()
    c.operator.train_conditions=1
    c.operator.condition_batch=1
    c.time_discretization=ConfigDict(dict(kind='bdf2',dt=.05))
    return c
