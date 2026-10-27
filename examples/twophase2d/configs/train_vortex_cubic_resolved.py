"""More periodic spatial features for the thin-interface accuracy control."""
from examples.twophase2d.configs.train_vortex_cubic_refined import get_config as refined_config


def get_config():
    c = refined_config()
    c.arch.spatial_modes = 8
    c.arch.hidden_dim = 96
    c.sample.Nt = 8
    c.sample.time_rule = 'open_stratified'
    return c
