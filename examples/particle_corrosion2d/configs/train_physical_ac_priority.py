"""Early AC sign violation motivates AC priority after gradient balancing."""
from examples.particle_corrosion2d.configs.train_physical import get_config as physical_config


def get_config():
    c=physical_config()
    c.train.iterations=5000
    c.loss.gradnorm_bias=(10.,1.,1.,1.,1.)
    return c
