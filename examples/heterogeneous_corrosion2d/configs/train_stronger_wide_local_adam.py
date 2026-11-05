"""Local AD comparison with the successful width96 Adam baseline."""
from examples.heterogeneous_corrosion2d.configs.train_stronger_wide import get_config as wider_config


def get_config():
    c = wider_config()
    c.nl.method = 'local_ad'
    c.loss.neumann_scales = (1., 1.)
    c.loss.gradnorm_bias = (1.,) * 7
    return c
