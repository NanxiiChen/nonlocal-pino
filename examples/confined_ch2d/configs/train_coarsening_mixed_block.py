"""Separate global fields with the frozen linear mixed Volterra block inverse."""
from ml_collections import ConfigDict
from examples.confined_ch2d.configs.train_coarsening_global_separate import get_config as baseline


def get_config():
    c=baseline();c.loss.phase_metric=ConfigDict(dict(kind='mixed_block',time_scale=c.phase.initial_time_scale))
    return c
