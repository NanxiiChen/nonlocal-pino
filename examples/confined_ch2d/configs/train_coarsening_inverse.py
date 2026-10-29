"""One global IC-increment network with Neumann inverse residual norm."""
from ml_collections import ConfigDict
from examples.confined_ch2d.configs.train_coarsening_eliminated_blend import get_config as baseline


def get_config():
    c=baseline();c.sample.temporal.curriculum=False
    c.sample.include_initial_time=True
    c.phase.lifting='logarithmic_increment';c.arch.initial_feature=True
    c.loss.phase_metric=ConfigDict(dict(kind='neumann_inverse'))
    return c
