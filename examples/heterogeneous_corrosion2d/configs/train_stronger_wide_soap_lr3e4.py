"""Conservative initial-LR SOAP control; all other settings remain identical."""
from examples.heterogeneous_corrosion2d.configs.train_stronger_wide_soap import get_config as soap_config


def get_config():
    c = soap_config()
    c.train.lr_init = 3e-4
    return c
