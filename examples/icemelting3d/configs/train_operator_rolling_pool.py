"""Matched spatial control: bounded background refreshed uniform core+halo."""
from examples.icemelting3d.configs.train_operator_fixed_pool import get_config as base_config


def get_config():
    c = base_config()
    c.sample.spatial.mode = 'rolling_pool'
    return c
