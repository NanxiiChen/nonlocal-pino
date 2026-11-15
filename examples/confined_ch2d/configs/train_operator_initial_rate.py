"""Opt-in initial-PDE-rate lift; matched full-coordinate conservative control."""
from ml_collections import ConfigDict
from examples.confined_ch2d.configs.train_operator_potential import get_config as baseline


def get_config():
    c=baseline()
    c.operator.initial_rate_lift=ConfigDict(dict(kind='exact_pde_rate',timescale='stable_graph_bound'))
    return c
