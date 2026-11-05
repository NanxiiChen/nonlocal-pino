"""Stronger spatial material contrast, with the latent constant mode fixed to0.

The initial pit, domain, governing coefficients, network and optimizer are
identical to the first pilot. This is a material prior change, not a pit-aware
directional forcing or an impermeable particle model.
"""
from ml_collections import ConfigDict

from examples.heterogeneous_corrosion2d.configs.train_base import get_config as base_config


def get_config():
    c = base_config()
    modes = c.material.modes
    c.material = ConfigDict(dict(minimum=0.5, maximum=4.0, modes=modes,
        coefficient_scale=(0.0,)+(1.3,)*(2*len(modes))))
    return c
