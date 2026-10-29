"""All spatial derivatives use finite-horizon operators; explicit research option."""
from examples.twophase2d.configs.train_vortex_conserved import get_config as conserved_config


def get_config():
    c = conserved_config()
    c.advection = 'nonlocal_flux'
    c.reference.path = '/root/autodl-tmp/data/twophase2d/results_periodic_vortex_nonlocal_flux.npz'
    return c
