"""Same global basis; instantaneous Neumann-inverse evolution residual."""
from ml_collections import ConfigDict
from examples.confined_ch2d.configs.train_coarsening_global_basis import get_config as baseline


def get_config():
    c=baseline();c.sample.include_initial_time=True
    c.loss.phase_metric=ConfigDict(dict(kind='neumann_inverse'))
    return c
