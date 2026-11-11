"""Matched spatial control: fixed32uniform core+halo sets, safe padding."""
from ml_collections import ConfigDict
from examples.icemelting3d.configs.train_operator_5k import get_config as base_config


def get_config():
    c = base_config()
    c.sample.spatial = ConfigDict(dict(mode='fixed_pool', max_nodes=1792,
        max_edges=1536, workers=1, prefetch=32, full_chunk_core=32768))
    return c
