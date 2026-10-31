"""Stronger geometry-response control: same IC/geometry/operator, longer CH time."""
from examples.confined_ch2d.configs.train_operator_mild import get_config as baseline


def get_config():
    c=baseline()
    c.physical.terminal_time=.75
    c.operator.dataset_path='/root/autodl-tmp/data/confined_ch2d/operator_geometry_longer_v1'
    return c
