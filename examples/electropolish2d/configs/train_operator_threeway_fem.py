"""Matched weak FEM with complete64-case local-FD monitoring."""
from ml_collections import ConfigDict
from examples.electropolish2d.configs.train_operator_threeway import get_config as baseline
from examples.electropolish2d.fem_weak import SPEC


def get_config():
    c = baseline()
    c.nl.method = 'fem_weak'
    c.operator.fem = ConfigDict(SPEC)
    return c
