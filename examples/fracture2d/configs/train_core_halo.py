"""Optional tip-refined core+halo route; the base configuration stays full."""
from examples.fracture2d.configs.train_base import get_config as base_config


def get_config():
    c = base_config()
    c.sample.mode = 'random'
    c.sample.boost = 18.
    c.sample.tip_boost = 36.
    c.sample.resample_every = 1
    c.output.root = '/root/autodl-tmp/tf-logs/fracture2d/runs/pinn_core_halo'
    c.output.event_dir = c.output.root + '/events'
    return c
