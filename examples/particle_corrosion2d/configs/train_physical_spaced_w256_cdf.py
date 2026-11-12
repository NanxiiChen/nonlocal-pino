"""Research control: probability-stratified cores without wall oversampling."""
from examples.particle_corrosion2d.configs.train_physical_spaced_w256 import get_config as baseline


def get_config():
    c = baseline()
    c.sample.core_draw = 'stratified_cdf'
    return c
