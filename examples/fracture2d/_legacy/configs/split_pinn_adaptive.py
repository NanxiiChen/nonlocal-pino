"""Experimental stable-checkpoint block stopping; uniform reference loading."""

from ml_collections import ConfigDict
from examples.fracture2d._legacy.configs.split_pinn_short_damage import get_config as base_config


def get_config():
    c = base_config()
    c.stagger.accept_on_first_passing_report = False
    c.inner_stop = ConfigDict()
    c.inner_stop.enabled = True
    c.inner_stop.check_every = 100
    c.inner_stop.min_updates = 200
    c.inner_stop.first_load_min_updates = 2000
    c.inner_stop.stable_checks = 5
    c.inner_stop.u_change_relative = 1e-4
    c.inner_stop.d_change_atol = 1e-6
    c.inner_stop.reaction_atol = 1e-8
    c.inner_stop.reaction_rtol = 1e-4
    c.inner_stop.objective_atol = 1e-8
    c.inner_stop.objective_rtol = 1e-4
    c.inner_stop.residual_atol = 1e-8
    c.inner_stop.residual_rtol = 1e-3
    c.inner_stop.mechanics_limit = 0.01
    c.inner_stop.damage_limit = 1e-3
    c.inner_stop.allow_stagnation_warning = True
    c.output.dir = "/root/autodl-tmp/tf-logs/fracture2d/pinn_short_damage_adaptive_20261009"
    c.output.event_dir = "/root/autodl-tmp/tf-logs/fracture2d/events/pinn_short_damage_adaptive_20261009"
    return c
