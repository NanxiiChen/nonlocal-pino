"""Cold SOAP control for the stronger width96 material-conditioned operator."""
from examples.heterogeneous_corrosion2d.configs.train_stronger_wide import get_config as wider_config


def get_config():
    c = wider_config()
    c.train.optimizer = 'soap'
    c.train.soap_b1 = 0.95
    c.train.soap_b2 = 0.9
    c.train.soap_precond_freq = 10
    c.train.soap_max_precond_dim = 256
    c.train.soap_precond_1d = False
    return c
