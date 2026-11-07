"""SOAP-only control on the fixed single-geometry optimization diagnostic."""
from examples.particle_corrosion2d.configs.train_physical_single import get_config as single_config


def get_config():
    c=single_config()
    c.train.optimizer='soap'
    return c
