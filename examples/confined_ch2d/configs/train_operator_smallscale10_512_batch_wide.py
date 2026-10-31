"""Capacity control: only widen64->96relative to the B16/Nt16 cold preset."""
from examples.confined_ch2d.configs.train_operator_smallscale10_512_batch import get_config as baseline


def get_config():
    c=baseline()
    c.arch.width=96
    return c
