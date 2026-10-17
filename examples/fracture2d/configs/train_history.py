"""From-zero coupled history-operator loading; no historical training inputs."""
from ml_collections import ConfigDict
from examples.fracture2d.configs.train_base import get_config as base_config


def get_config():
    c = base_config()
    c.arch.type = 'history_operator'
    c.history = ConfigDict(dict(coarse_cells=8, fine_cells=32, fine_tokens=64,
        width=32, heads=2, point_depth=2, query_depth=2, head_depth=2,
        replay_capacity=128, replay_batch=1, replay_fraction=.25, replay_enabled=True,
        replay_every=1, current_only_steps=0))
    c.train.learning_rate = 1e-3
    c.train.minimum_rate = 1e-3
    c.output.root = '/root/autodl-tmp/tf-logs/fracture2d/runs/pinn_history_replay'
    c.output.event_dir = c.output.root + '/events'
    return c
