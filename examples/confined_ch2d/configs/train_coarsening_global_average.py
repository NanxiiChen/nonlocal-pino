"""One global network and time-average PDE integral residual R(t)/t."""
from examples.confined_ch2d.configs.train_coarsening_global_basis import get_config as baseline


def get_config():
    c=baseline();c.loss.phase_metric.normalization='time_average'
    return c
