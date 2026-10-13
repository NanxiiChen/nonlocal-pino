"""User-authorized full warning continuation; reference discrepancies do not gate."""

from examples.fracture2d.configs.split_pinn_short_damage import get_config as base_config


def get_config():
    c = base_config()
    c.reporting.comparison_gate = False
    c.stagger.accept_on_first_passing_report = False
    c.data.end_state_index = 75
    c.output.fork_from = "/root/autodl-tmp/tf-logs/fracture2d/pinn_short_damage_first2_20261009"
    c.output.fork_after_index = 53
    c.output.dir = "/root/autodl-tmp/tf-logs/fracture2d/pinn_short_full_warning_from53_20261010"
    c.output.event_dir = "/root/autodl-tmp/tf-logs/fracture2d/events/pinn_short_full_warning_from53_20261010"
    return c
