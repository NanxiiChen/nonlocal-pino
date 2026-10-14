"""Identical cold all-node control, with 1e-4 early load increments."""
from examples.fracture2d.configs.full_damage_load_coarse import get_config as base


def get_config():
    c = base()
    c.full_loading.case = 'fine'
    c.full_loading.pre_step = 1e-4
    return c
