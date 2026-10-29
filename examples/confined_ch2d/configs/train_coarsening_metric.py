"""IC/time-coordinate control with frozen stiff-operator residual metric."""
from ml_collections import ConfigDict
from examples.confined_ch2d.configs.train_coarsening_logtime import get_config as baseline


def get_config():
    c=baseline()
    c.loss.phase_metric=ConfigDict(dict(kind='biharmonic_resolvent',time_scale=.005))
    return c
