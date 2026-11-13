"""Local AD control; original two explicit normalized wall losses."""
from examples.particle_corrosion2d.configs.train_fem_comparison_nonlocal import get_config as baseline


def get_config():
    c = baseline()
    c.nl.method = 'local_ad'
    c.loss.gradnorm_bias = tuple(c.loss.gradnorm_bias)+(1., 1.)
    return c
