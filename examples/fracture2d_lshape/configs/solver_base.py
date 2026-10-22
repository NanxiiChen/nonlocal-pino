"""Matched direct P1 L-panel reference with strict field-change stopping."""
from examples.fracture2d_lshape.configs.train_base import get_config as neural_config


def get_config():
    c = neural_config()
    del c.train.rate_control
    del c.loading.damage_stop
    c.loading.prefix_outer_steps = 1024
    c.loading.outer_steps = 1024
    c.output.root = '/root/autodl-tmp/data/fracture2d_lshape/fem_reference'
    c.output.event_dir = '/root/autodl-tmp/tf-logs/fracture2d_lshape/fem_reference/events'
    c.output.reference_root = ''
    return c
