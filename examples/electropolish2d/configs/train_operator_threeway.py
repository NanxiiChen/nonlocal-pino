"""Nonlocal three-method control with common read-only local-FD monitoring."""
from ml_collections import ConfigDict
from examples.electropolish2d.configs.train_operator_rolling_nonlocal import get_config as baseline


def get_config():
    c = baseline()
    c.operator.online.input_evaluation = 'host_float64_shared'
    c.operator.comparison_monitor = ConfigDict(dict(
        inputs_root='/root/autodl-tmp/data/electropolish2d/operator_test64_20261028',
        reference_root='/root/autodl-tmp/data/electropolish2d/operator_local_test64_20261029',
        indices=tuple(range(64)), time_count=21))
    return c
