"""Random-ellipsoid operator AD control; physics and cold10k recipe unchanged."""
from ml_collections import ConfigDict
from examples.icemelting3d.configs.train_operator_compare_nonlocal import get_config as baseline


def get_config():
    c = baseline()
    c.operator.spatial_discretization = 'local_ad'
    c.operator.local_ad = ConfigDict(dict(boundary_points=96, boundary_weight=1.,
        boundary_norm='physical_normal_mse', initial_derivatives='analytic_unique_projection'))
    return c
