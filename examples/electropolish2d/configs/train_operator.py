"""Recommended rolling-pool Electropolish PINO with 16 shared times and width 96."""
from examples.electropolish2d.configs.train_operator_rolling_pool import get_config as baseline


def get_config():
    c = baseline()
    c.arch.width = 96
    c.sample.Nt = 16
    c.train.iterations = 10_000
    return c
