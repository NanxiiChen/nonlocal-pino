"""Capacity control: only widen64->96relative to the matched SOAP preset."""
from examples.confined_ch2d.configs.train_operator_smallscale10_512_soap import get_config as baseline


def get_config():
    c=baseline()
    c.arch.width=96
    return c
