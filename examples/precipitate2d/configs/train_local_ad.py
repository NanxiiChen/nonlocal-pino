"""Matched control: point-network AD Hessians for both mixed phase fields."""
from examples.precipitate2d.configs.train_base import get_config as base_config


def get_config():
    c = base_config()
    c.nl.method = 'local_ad'
    return c
