"""Fixed 2048-entry core+halo comparison against the full-domain baseline."""
from examples.fracture2d.configs.train_base import get_config as base_config


def get_config():
    c = base_config()
    c.sample.mode = 'random'
    c.output.root = '/root/autodl-tmp/tf-logs/fracture2d/pinn_core+halo'
    c.output.event_dir = '/root/autodl-tmp/tf-logs/pinn_core+halo'
    return c
