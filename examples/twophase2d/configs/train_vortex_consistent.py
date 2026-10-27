"""Original output/residual with consistent conservative nodal advection."""
from examples.twophase2d.configs.train_vortex_refined import get_config as refined_config


def get_config():
    c = refined_config()
    c.phase_mass = 'exact_grid_mean'
    del c.loss.mass_weight
    c.advection = 'spectral_flux'
    c.sample.Nt = 8
    c.sample.time_rule = 'open_stratified'
    return c
