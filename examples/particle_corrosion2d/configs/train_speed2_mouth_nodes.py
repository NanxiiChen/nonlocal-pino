"""Twice-speed dynamics with BC losses at every discrete reservoir node."""
from examples.particle_corrosion2d.configs.train_speed2 import get_config as speed2_config


def get_config():
    c = speed2_config()
    c.operator.mouth_sampling = 'mesh_dirichlet'
    return c
