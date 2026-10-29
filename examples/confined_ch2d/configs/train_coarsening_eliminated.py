"""Exact native chemical relation and stiff-resolvent metric; one learned field."""
from ml_collections import ConfigDict
from examples.confined_ch2d.configs.train_coarsening_metric import get_config as baseline


def get_config():
    c=baseline()
    c.formulation=ConfigDict(dict(chemical='eliminated'))
    del c.loss.chemical_weight
    return c
