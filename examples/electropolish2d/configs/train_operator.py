"""Recommended physics-only NO: online B8/5000 Adam, image-only condition."""
from examples.electropolish2d.configs.train_operator_online import get_config as online_config


def get_config():
    c = online_config()
    c.arch.local_input = 'zero'
    return c
