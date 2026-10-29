"""Diagnostic two-fixed-input control; uniform-time original physical loss."""
from examples.electropolish2d.configs.train_operator_single import get_config as single_config


def get_config():
    c = single_config()
    c.operator.training_indices = (0, 1)
    c.operator.condition_batch = 2
    return c
