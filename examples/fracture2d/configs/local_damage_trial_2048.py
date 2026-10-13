"""Larger outer budget for the separately identified local-mechanics trial."""
from examples.fracture2d.configs.local_damage_trial import get_config as base


def get_config():
    c = base()
    c.outer_stop.max_sweeps = 2048
    c.trial.output_base = '/root/autodl-tmp/tf-logs/fracture2d/local_damage_fresh_budget2048_20261011'
    return c
