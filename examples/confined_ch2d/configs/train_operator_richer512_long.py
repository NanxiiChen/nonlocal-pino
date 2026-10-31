"""Matched cold20000-step budget; retain the original learning-rate schedule."""
from examples.confined_ch2d.configs.train_operator_richer512 import get_config as baseline


def get_config():
    c=baseline()
    c.train.iterations=20000
    return c
