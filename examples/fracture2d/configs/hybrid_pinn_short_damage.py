"""Direct nonlocal mechanics with one FP64 nonlocal AT2 damage network."""

from examples.fracture2d.configs.split_pinn_short_damage import get_config as base_config


def get_config():
    c = base_config()
    c.train.mechanics_solver = "direct"
    del c.train.mechanics_objective
    del c.stagger.mechanics_steps
    c.stagger.accept_on_first_passing_report = False
    c.reporting.comparison_gate = False
    c.output.dir = "/root/autodl-tmp/tf-logs/fracture2d/hybrid_short_damage_first2_20261010"
    c.output.event_dir = "/root/autodl-tmp/tf-logs/fracture2d/events/hybrid_short_damage_first2_20261010"
    return c
