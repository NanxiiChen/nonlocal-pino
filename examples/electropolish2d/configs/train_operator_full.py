"""Matched full-coordinate control: PDE residual on every free graph node."""
from examples.electropolish2d.configs.train_operator_fixed_pool import get_config as baseline


def get_config():
    c = baseline()
    c.sample.spatial.mode = 'full'
    return c
