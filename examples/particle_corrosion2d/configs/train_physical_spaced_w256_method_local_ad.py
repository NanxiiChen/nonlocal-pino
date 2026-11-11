"""Local-AD counterpart; identical rolling inputs and two explicit wall losses."""
from examples.particle_corrosion2d.configs.train_physical_spaced_w256_method_nonlocal import (
    get_config as nonlocal_config,
)


def get_config():
    c = nonlocal_config()
    c.nl.method = 'local_ad'
    c.loss.gradnorm_bias = tuple(c.loss.gradnorm_bias)+(1., 1.)
    return c
