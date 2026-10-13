"""Bounded own-history PINN comparison, not a convergence budget."""
from examples.fracture2d.configs.local_damage_trial import get_config as base


def get_config():
    c = base()
    c.trial.damage_solver = 'pinn'
    c.trial.output_base = '/root/autodl-tmp/tf-logs/fracture2d/local_damage_pinn_smoke4_20261011'
    c.outer_stop.max_sweeps = 4
    c.data.end_state_index = 2
    return c
