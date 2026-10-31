"""Global Fourier operator with an explicit expanding log-time training measure."""
from ml_collections import ConfigDict
from examples.confined_ch2d.configs.train_operator_smallscale10_512_fourier import get_config as baseline


def get_config():
    c = baseline()
    c.sample.temporal = ConfigDict(dict(kind='log_uniform', time_scale=.05,
        curriculum=True, initial_horizon=.15, warmup_iterations=8000))
    return c
