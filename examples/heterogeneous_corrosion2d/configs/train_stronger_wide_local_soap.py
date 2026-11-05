"""Matched local spatial-AD/SOAP control with explicit reflecting-wall BCs."""
from examples.heterogeneous_corrosion2d.configs.train_stronger_wide_soap import get_config as soap_config


def get_config():
    c = soap_config()
    c.nl.method = 'local_ad'
    c.loss.neumann_scales = (1., 1.)
    c.loss.gradnorm_bias = (1.,) * 7
    return c
