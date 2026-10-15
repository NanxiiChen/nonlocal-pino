"""Experimental user-proposed coarse/fine loading plus adaptive inner blocks.

This samples a different cumulative-history path; it is not validated as
equivalent to the uniform direct/PINN trajectory.
"""

from examples.fracture2d._legacy.configs.split_pinn_adaptive import get_config as adaptive_config


def get_config():
    c = adaptive_config()
    c.data.reference_indices = [10, 20, 30, 40, 50, *range(51, 76)]
    c.data.end_state_index = len(c.data.reference_indices)
    c.output.dir = "/root/autodl-tmp/tf-logs/fracture2d/pinn_short_damage_coarse_fine_20261009"
    c.output.event_dir = "/root/autodl-tmp/tf-logs/fracture2d/events/pinn_short_damage_coarse_fine_20261009"
    return c
