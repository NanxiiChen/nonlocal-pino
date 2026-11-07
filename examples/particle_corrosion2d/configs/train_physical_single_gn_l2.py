"""Euclidean gradient balancing refreshed every update; original PDE/outputs."""
from examples.particle_corrosion2d.configs.train_physical_single import get_config as single_config


def get_config():
    c=single_config()
    c.loss.gradnorm_norm='l2'
    c.train.gradnorm_update_every=1
    return c
