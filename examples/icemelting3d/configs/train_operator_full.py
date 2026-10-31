"""Exact full128^3mean loss, accumulated spatial chunks before each update."""
from examples.icemelting3d.configs.train_operator_fixed_pool import get_config as base_config


def get_config():
    c = base_config()
    c.sample.spatial.mode = 'full'
    return c
