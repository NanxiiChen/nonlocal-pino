"""Unit decoder gain on the simpler linear-time stable-part strategy."""
from examples.confined_ch2d.configs.train_operator_smallscale10_512_stable import get_config as baseline


def get_config():
    c=baseline()
    c.arch.output_scale=1.
    return c
