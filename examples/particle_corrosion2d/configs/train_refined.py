"""Joint quadrature/horizon and collocation-coverage refinement control."""
from examples.particle_corrosion2d.configs.train_wide import get_config as wide_config


def get_config():
    c=wide_config()
    c.mesh.spacing=.01
    c.nl.delta=.020
    c.nl.max_halo=6144
    c.sample.K_pool=32
    return c
