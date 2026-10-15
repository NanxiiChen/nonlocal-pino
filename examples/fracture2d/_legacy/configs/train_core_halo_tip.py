"""Tip-refined fixed pool; batch hold interval remains an explicit override."""
from examples.fracture2d.configs.train_core_halo import get_config as core_halo_config


def get_config():
    c = core_halo_config()
    c.sample.boost = 18.
    c.sample.tip_boost = 36.
    c.sample.resample_every = 1
    c.output.root = '/root/autodl-tmp/tf-logs/fracture2d/pinn_core+halo_tip'
    c.output.event_dir = '/root/autodl-tmp/tf-logs/pinn_core+halo_tip'
    return c
