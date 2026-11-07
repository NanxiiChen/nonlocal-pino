"""CH priority after gradient balancing; no physical or raw-residual pre-scale."""
from examples.particle_corrosion2d.configs.train_physical_single import get_config as single_config


def get_config():
    c=single_config()
    c.loss.gradnorm_bias=(1.,10.,1.,1.,1.)
    return c
