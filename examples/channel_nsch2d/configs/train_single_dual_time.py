"""Cold general startup-resolution control; same feature width and physical PDE."""
from ml_collections import ConfigDict
from examples.channel_nsch2d.configs.train_single_conservative import get_config as baseline


def get_config():
    c=baseline();c.sample.Nx=192;c.sample.Ny=80;c.train.iterations=10000
    c.time_embedding=ConfigDict(dict(kind='linear_log',scale=c.lifting.time_scale))
    return c
