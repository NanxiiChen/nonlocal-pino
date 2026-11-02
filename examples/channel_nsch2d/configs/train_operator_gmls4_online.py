"""Untrained online extension of the explicit higher-order operator control."""
from examples.channel_nsch2d.configs.train_operator_conservative_online import get_config as baseline


def get_config():
    c=baseline();c.nl.degree=4;c.nl.neighbors=48;c.nl.weight_exponent=12.
    return c
