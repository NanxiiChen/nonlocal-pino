"""Matched local AD with complete64-case local-FD monitoring."""
from examples.electropolish2d.configs.train_operator_threeway import get_config as baseline


def get_config():
    c = baseline()
    c.nl.method = 'local_ad'
    c.loss.gradnorm_bias = tuple(c.loss.gradnorm_bias)+(1.,)
    c.loss.neumann_scale = 1.
    return c
