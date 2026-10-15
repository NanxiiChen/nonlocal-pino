"""Explicit admissible initial guess; preserve the six ready hybrid predecessors."""
from examples.fracture2d._legacy.configs.local_damage_corrected import get_config as base


def get_config():
    c = base()
    c.predictor.initial_projection = 'unit_interval'
    c.trial.fork_source = '/root/autodl-tmp/tf-logs/fracture2d/local_damage_corrected_20261011'
    c.trial.fork_ready_loads = 6
    c.trial.output_base = '/root/autodl-tmp/tf-logs/fracture2d/local_damage_corrected_projected_20261011'
    return c
