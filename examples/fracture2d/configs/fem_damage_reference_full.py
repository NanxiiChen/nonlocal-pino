"""Matching full FE direct reference; run before train_history_fem_full."""
from examples.fracture2d.configs.train_history_fem_full import get_config as neural_config


def get_config():
    c = neural_config()
    c.output.root = '/root/autodl-tmp/tf-logs/fracture2d/runs/fem_damage_reference_full'
    c.output.event_dir = c.output.root + '/events'
    c.output.reference_root = ''
    return c
