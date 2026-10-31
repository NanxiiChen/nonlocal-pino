"""Matched background core+halo control; deterministic fresh sets after32."""
from examples.electropolish2d.configs.train_operator_fixed_pool import get_config as baseline


def get_config():
    c = baseline()
    c.sample.spatial.mode = 'rolling_pool'
    return c
