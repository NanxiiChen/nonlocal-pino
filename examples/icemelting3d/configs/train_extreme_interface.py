"""Opt-in thin-interface/horizon study; canonical presets remain unchanged."""
from ml_collections import ConfigDict
from examples.icemelting3d.configs.train_thin_interface import get_config as pilot


def get_config():
    c = pilot()
    c.physical.eps_star = .00375
    c.train.iterations = 10000
    c.train.lr_decay_every = 2000
    c.nl.normalization = 'raw'
    c.experiment.kind = 'extreme_interface'
    c.experiment.quadrature_directions = 256
    c.experiment.quadrature_radial = 4096
    c.experiment.radial_directions = 32
    c.experiment.radial_points = 4096
    c.reference = ConfigDict(dict(grids=(8193, 16385), outer_radius=.9,
        rtol=1e-9, atol=1e-10, max_step=.0025))
    return c
