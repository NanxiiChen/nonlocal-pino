"""Validated small-load speed trial from learned state 21; same physical model."""

from examples.fracture2d._legacy.configs.split_pinn_fine import get_config as fine_config


def get_config():
    c = fine_config()
    c.train.operator_mode = "analytic_padded_damage"
    c.stagger.mechanics_steps = 200
    c.stagger.damage_steps = 500
    c.data.end_state_index = 23
    c.output.fork_from = "/root/autodl-tmp/tf-logs/fracture2d/pinn_split_speed_baseline21_20261002"
    c.output.fork_after_index = 21
    c.output.dir = "/root/autodl-tmp/tf-logs/fracture2d/pinn_split_fast_from21_20261002"
    # Fewer updates are validated only in this small-load segment. Retain the
    # full-budget config for refinement if damage/reaction accuracy deteriorates.
    return c
