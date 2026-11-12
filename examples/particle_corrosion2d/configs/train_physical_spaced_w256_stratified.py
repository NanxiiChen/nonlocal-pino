"""Research control: 64 near-wall and 192 bulk probability-stratified cores."""
from ml_collections import ConfigDict
from examples.particle_corrosion2d.configs.train_physical_spaced_w256 import get_config as baseline


def get_config():
    c = baseline()
    c.sample.core_draw = 'stratified_cdf'
    c.sample.core_wall = ConfigDict(dict(distance=c.nl.delta, count=64))
    return c
