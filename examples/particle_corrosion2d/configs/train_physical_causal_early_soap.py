"""Original strong residuals and early sampling, changing only Adam to SOAP."""
from examples.particle_corrosion2d.configs.train_physical_causal_early import get_config as early_config


def get_config():
    c=early_config()
    c.train.optimizer='soap'
    return c
