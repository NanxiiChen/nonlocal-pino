"""Convergence-gated hybrid trial from the same warning .0054 predecessor."""

from ml_collections import ConfigDict
from examples.fracture2d._legacy.configs.hybrid_pinn_swish_fourier_128 import get_config as base_config


def get_config():
    c = base_config()
    c.outer_stop = ConfigDict()
    c.outer_stop.enabled = True
    c.outer_stop.min_sweeps = 3
    c.outer_stop.max_sweeps = 128
    c.outer_stop.consecutive_passes = 3
    c.outer_stop.inner_AT2_relative_limit = 1e-3
    c.outer_stop.q_change_limit = 1e-4
    c.outer_stop.H_change_limit = 1e-4
    c.outer_stop.reaction_change_limit = 1e-4
    c.outer_stop.scale_floor = 1e-12
    # This is a conditional same-parent diagnostic, not a validated history path.
    c.data.end_state_index = 8  # ordinal8/reference55 is .0055.
    c.output.fork_from = "/root/autodl-tmp/tf-logs/fracture2d/hybrid_swish_fourier_128x6_from0048_first2_20261010"
    c.output.fork_after_index = 7
    c.output.dir = "/root/autodl-tmp/tf-logs/fracture2d/hybrid128_stagger_gated0055_20261010"
    c.output.event_dir = "/root/autodl-tmp/tf-logs/fracture2d/events/hybrid128_stagger_gated0055_20261010"
    return c
