"""One global network, all times jointly trained; no frozen intermediate states."""
from ml_collections import ConfigDict
from examples.confined_ch2d.configs.train_coarsening_eliminated_blend import get_config as baseline


def get_config():
    c=baseline()
    c.sample.temporal.curriculum=False
    c.loss.phase_metric=ConfigDict(dict(kind='space_time_resolvent'))
    return c
