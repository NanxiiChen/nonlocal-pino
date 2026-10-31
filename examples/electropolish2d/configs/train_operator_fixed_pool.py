"""Matched fixed32-entry core+halo control with safe fixed-shape padding."""
from ml_collections import ConfigDict
from examples.electropolish2d.configs.train_operator import get_config as baseline


def get_config():
    c = baseline()
    c.sample.spatial = ConfigDict(dict(mode='fixed_pool', max_nodes=3328,
        max_edges=3072, workers=1, prefetch=32))
    return c
