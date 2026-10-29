"""Matched control: replace phase nonlocal diffusion with Fourier -Delta."""
from examples.precipitate2d.configs.train_base import get_config as base_config


def get_config():
    c = base_config()
    c.nl.method = 'local_spectral'
    return c
