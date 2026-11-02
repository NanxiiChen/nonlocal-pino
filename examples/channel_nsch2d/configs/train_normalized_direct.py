"""Preserved normalized-lifting, direct-velocity single-geometry failure control."""
from examples.channel_nsch2d.configs.train_base import get_config as baseline


def get_config():
    c=baseline()
    del c.velocity
    c.arch.width=64;c.arch.decoder_depth=2;c.arch.spatial_features=32
    c.operator.train_conditions=1;c.operator.condition_batch=1
    return c
