"""Full-field width256/online50k control with matched positive-time PDE measure."""
from ml_collections import ConfigDict
from examples.confined_ch2d.configs.train_operator import get_config as baseline


def get_config():
    c=baseline()
    c.operator.spatial_discretization='nonlocal'
    c.train.autotune_level=0
    # The fixed ambient IC need not satisfy the newly sampled Neumann walls.
    c.loss.initial_residual_fraction=0.
    c.operator.comparison_reference=ConfigDict(dict(kind='local_fem_mixed_p2',
        path='/root/autodl-tmp/data/confined_ch2d/operator_threeway_local_20261114_v2'))
    return c
