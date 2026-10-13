"""Separate own-zero control using the verified baseline's original inner gate."""
from examples.fracture2d.configs.compact_damage_loading_scaled import get_config as base


def get_config():
    c = base()
    c.loading.output_dir = '/root/autodl-tmp/tf-logs/fracture2d/compact_damage_loading_original_gates_20261011'
    c.loading.lbfgs.residual_limit = c.outer_stop.inner_AT2_relative_limit
    return c
