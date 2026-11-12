"""Controlled sphere-melting thickness study; production defaults are unchanged."""
from ml_collections import ConfigDict
from examples.icemelting3d.configs.train_base import get_config as baseline


def get_config():
    c = baseline()
    c.sample.K_core = 256
    c.sample.Nt = 16
    c.train.iterations = 5000
    c.train.lr_decay_rate = .5
    c.log.eval_every = 1000
    c.experiment = ConfigDict(dict(kind='thin_interface',
        reference_path='', boundary_points=48, boundary_weight=1.,
        normalization='divide_by_N3', reference_grids=(384, 512),
        reference_dt=.00125, reference_form='latent', snapshots=9, evaluation_axis=64,
        evaluation_chunk=8192, radial_points=384, radial_directions=24))
    return c
