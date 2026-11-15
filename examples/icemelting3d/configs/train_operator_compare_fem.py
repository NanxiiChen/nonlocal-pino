"""Matched P1-tetrahedral weak-operator PINO; native refs remain diagnostics."""
from examples.icemelting3d.configs.train_operator_compare_nonlocal import get_config as baseline


def get_config():
    c = baseline()
    c.operator.spatial_discretization = 'fem_p1_lumped'
    return c
