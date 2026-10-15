"""Extend the original-gate own-zero neural prefix through all75 loads."""
from examples.fracture2d._legacy.configs.compact_damage_loading_original_gates import get_config as base


def get_config():
    c = base()
    c.data.end_state_index = 75
    return c
