"""Moderate SI kinetics with longer evolution and visible particle interaction."""
from examples.particle_corrosion2d.configs.train_physical import get_config as physical_config
from examples.particle_corrosion2d.dimensional import derive_config


def get_config():
    c=physical_config()
    c.physical.L=2e-9
    c.nondim.T_c=350.0
    return derive_config(c)
