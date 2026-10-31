"""Wider space/time Fourier inputs on the unchanged fixed10mode physical case."""
from examples.confined_ch2d.configs.train_operator_smallscale10_512_soap import get_config as baseline


def get_config():
    c=baseline()
    c.arch.spatial_features=64
    c.arch.spatial_scale=2.
    c.arch.time_features=32
    c.train.soap_max_precond_dim=256
    return c
