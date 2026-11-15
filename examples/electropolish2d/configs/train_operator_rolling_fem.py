"""Matched rolling P1/lumped weak-residual PINO; same original five losses."""
from ml_collections import ConfigDict
from examples.electropolish2d.configs.train_operator_rolling_nonlocal import get_config as baseline
from examples.electropolish2d.fem_weak import SPEC


def get_config():
    c = baseline()
    c.nl.method = 'fem_weak'
    c.operator.fem = ConfigDict(SPEC)
    return c
