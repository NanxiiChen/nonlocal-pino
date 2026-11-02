"""Cold20k BC-coverage control; original model, quadrature and joint PDE losses."""
from ml_collections import ConfigDict
from examples.channel_nsch2d.configs.train_operator_width128_gmls_first_online import get_config as baseline


def get_config():
    c=baseline()
    c.boundary_sampling=ConfigDict(dict(kind='gauss_legendre',points_per_side=192,
        encoder='unchanged_midpoints'))
    return c
