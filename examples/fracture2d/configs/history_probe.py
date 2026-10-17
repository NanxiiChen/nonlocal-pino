"""Small fixed-history point-encoder experiment; accepted defaults stay unchanged."""
from ml_collections import ConfigDict
from examples.fracture2d.configs.train_base import get_config as base_config


def get_config():
    c = base_config()
    c.history = ConfigDict(dict(
        source_root='/root/autodl-tmp/tf-logs/fracture2d/pinn_mlp_baseline',
        loads=[.0050, .0051, .0052], held_out_load=.0053,
        coarse_cells=8, fine_cells=32, fine_tokens=64,
        width=32, heads=2, point_depth=2, query_depth=2, head_depth=2,
        replay_capacity=128, replay_batch=1, replay_fraction=.25,
        steps_per_snapshot=256))
    c.train.learning_rate = 1e-3
    c.train.minimum_rate = 1e-3
    c.output.root = '/root/autodl-tmp/tf-logs/fracture2d/runs/history_probe'
    c.output.tensorboard = False
    return c
