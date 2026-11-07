"""Six separated inclusions outside the initial diffuse pit interface."""
from examples.particle_corrosion2d.configs.train_physical_moderate import get_config as moderate_config


def get_config():
    c=moderate_config()
    c.particles.count=6
    c.particles.pit_clearance=.075
    c.particles.particle_clearance=.055
    c.particles.near_pit_distance=.16
    c.particles.minimum_near_pit_count=2
    return c
