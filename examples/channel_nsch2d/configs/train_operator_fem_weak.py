"""Full-coordinate cold20k FE weak control; preserves the accepted network/physics."""
from ml_collections import ConfigDict
from examples.channel_nsch2d.configs.train_operator_random_walls_width192_batch8_time18_nt36_cold20k import get_config as baseline


def get_config():
    c = baseline()
    c.nl = ConfigDict(dict(method='fem_weak_p2p1', quadrature_order=6,
                          residual_metric='mass_diagonal_dual', geometry='isoparametric_p2'))
    del c.derivatives
    del c.phase_transport
    return c
