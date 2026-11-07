"""Resolve fast kinetics through the existing decoder coordinate; no IC map."""
from ml_collections import ConfigDict
from examples.particle_corrosion2d.configs.train_physical_causal_focus import get_config as focus_config


def get_config():
    c=focus_config()
    c.arch.time_coordinate=ConfigDict(dict(kind='logarithmic',
        time_scale=1/(2*c.physical.kappa*c.physical.n_ac2)))
    return c
