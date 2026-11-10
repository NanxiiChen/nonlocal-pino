"""Explicit Electropolish time injection study with the original operator recipe."""
from examples.electropolish2d.configs.train_operator_5k import get_config as operator_config


def get_config():
    c = operator_config()
    c.arch.temporal_query = 'query_film'
    return c
