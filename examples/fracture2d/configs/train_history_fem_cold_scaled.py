"""Accepted cold FEM-energy history baseline with damage-scaled stopping."""
from ml_collections import ConfigDict
from examples.fracture2d.configs.train_history_fem_cold import get_config as cold_config


def get_config():
    c = cold_config()
    c.loading.damage_stop = ConfigDict(dict(mode='scaled_max', offset=.01, factor=.001))
    c.train.rate_control.mode = 'progress_blocks'
    c.train.rate_control.max_blocks_per_load = 4
    c.train.rate_control.minimum_reaction_decrease = .01
    c.output.root = '/root/autodl-tmp/tf-logs/fracture2d/pinn_history_fem_scaled_stop_baseline'
    c.output.event_dir = c.output.root + '/events'
    return c
