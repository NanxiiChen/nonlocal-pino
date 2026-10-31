"""Matched cold SOAP control; retain architecture, budget, LR and geometry bank."""
from examples.confined_ch2d.configs.train_operator_smallscale10_512 import get_config as baseline


def get_config():
    c=baseline()
    c.train.optimizer='soap'
    return c
