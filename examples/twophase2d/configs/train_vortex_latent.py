"""Invertible unbounded phase map; test exact inverse-map residual preconditioning."""
from ml_collections import ConfigDict
from examples.twophase2d.configs.train_vortex_residual import get_config as residual_config


def get_config():
    c = residual_config()
    c.phase = ConfigDict(dict(map='leaky_tanh', leak=0.1, inverse_steps=8))
    c.residual.mode = 'map_both'
    return c
