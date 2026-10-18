"""Same P1 damage operator/loads/stagger as train_history_fem, direct damage."""
from examples.fracture2d.configs.train_history_fem import get_config as neural_config


def get_config():
    c = neural_config()
    c.output.root = '/root/autodl-tmp/tf-logs/fracture2d/runs/fem_damage_reference_pilot'
    c.output.event_dir = c.output.root + '/events'
    return c
