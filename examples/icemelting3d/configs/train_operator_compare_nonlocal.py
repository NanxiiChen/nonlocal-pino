"""Paired cold10k spatial control; production physics and rolling sampling."""
from examples.icemelting3d.configs.train_operator import get_config as baseline


def get_config():
    c = baseline()
    c.operator.spatial_discretization = 'nonlocal'
    c.operator.monitor_split = 'test'
    c.operator.test_conditions = 64
    c.operator.dataset_path = '/root/autodl-tmp/data/icemelting3d/operator_ellipsoids_test64_20261114_v1'
    c.train.autotune_level = 0
    return c
