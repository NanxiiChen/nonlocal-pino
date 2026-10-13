"""Local FEM / nonlocal AT2 isolation, retaining the current fine-cloud BVP."""
from ml_collections import ConfigDict
from examples.fracture2d.configs.hybrid_pinn_swish_fourier_128 import get_config as base


def get_config():
    c = base()
    c.trial = ConfigDict()
    c.trial.damage_solver = 'direct'
    c.trial.geometry_dir = '/root/autodl-tmp/tf-logs/fracture2d/local_damage_geometry_20261011'
    c.trial.output_base = '/root/autodl-tmp/tf-logs/fracture2d/local_damage_from0048_20261011_v2'
    c.data.end_state_index = 28
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
    return c
