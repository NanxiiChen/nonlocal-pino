"""Cold capacity control: width256 with 256 sin/cos position features."""
from examples.particle_corrosion2d.configs.train_physical_spaced import (
    get_config as spaced_config,
)


def get_config():
    c = spaced_config()
    c.arch.width = 256
    c.arch.spatial_features = 128
    return c
