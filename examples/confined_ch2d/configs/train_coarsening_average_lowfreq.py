"""Retained phase-only integral R/t experiment with spatial Fourier modes1..5."""
from examples.confined_ch2d.configs.train_coarsening_global_average import get_config as baseline


def get_config():
    c=baseline();c.arch.spatial_modes=5
    return c
