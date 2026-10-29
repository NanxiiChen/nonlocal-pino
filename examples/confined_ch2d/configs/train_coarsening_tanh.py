"""Phase-only tanh output/mass map, original instantaneous PDE residual."""
from ml_collections import ConfigDict
from examples.confined_ch2d.configs.train_coarsening_global_strong import get_config as baseline


def get_config():
    c=baseline();c.phase.output_map=ConfigDict(dict(kind='tanh',amplitude=1.1))
    c.loss.residual_transform=ConfigDict(dict(kind='none'))
    return c
