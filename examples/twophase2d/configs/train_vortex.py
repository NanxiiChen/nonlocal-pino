"""Periodic reversed-vortex deformation; explicit variant, not paper reproduction."""
from ml_collections import ConfigDict
from examples.twophase2d.configs.train_base import get_config as base_config


def get_config():
    c = base_config()
    c.physical = ConfigDict(dict(epsilon=0.02, mobility=0.0001,
        terminal_time=2.0, velocity=ConfigDict(dict(
            kind='reversed_vortex', amplitude=2.0, period=2.0))))
    c.initial.center = (0.5, 0.75)
    c.initial.level = 0.2
    c.initial.width = 0.06
    c.sample.Nx = c.sample.Ny = 64
    c.nl.delta = 0.04
    c.arch.hidden_dim = 64
    c.arch.num_layers = 4
    c.reference.path = '/root/autodl-tmp/data/twophase2d/results_periodic_vortex.npz'
    c.reference.local_grid = 128
    c.reference.refined_local_grid = 256
    return c
