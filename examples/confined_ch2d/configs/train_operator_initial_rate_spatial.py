"""Same exact initial-rate experiment with rolling wall/bulk core groups."""
from ml_collections import ConfigDict
from examples.confined_ch2d.configs.train_operator_potential_spatial import get_config as baseline


def get_config():
    c=baseline()
    c.operator.initial_rate_lift=ConfigDict(dict(kind='exact_pde_rate',timescale='stable_graph_bound'))
    return c
