"""Recompute the successful fine prefix with fresh network and Adam state."""
from examples.fracture2d.configs.full_damage_load_coarse import get_config as base


def get_config():
    c = base()
    a = c.full_loading
    a.suite_root = '/root/autodl-tmp/tf-logs/fracture2d/cold_full_pinn_loading_0060_20261014/prefix'
    # Existing cold driver uses coarse as its first independently initialized case ID.
    # Actual increments are fine throughout; this is not a coarse-step simulation.
    a.case = 'coarse'
    a.pre_step = 1e-4
    a.end_load = .0052
    a.critical_load = .0052
    a.critical_outer_cap = 128
    a.checkpoint_sweeps = [1, 8, 32, 64, 128]
    a.oracle_loads = [.0050, .0052]
    return c
