"""Known-IC latent increments and exact reservoir data; two original PDE losses."""
from ml_collections import ConfigDict
from examples.particle_corrosion2d.configs.train_physical_hard_log_l2 import get_config as log_config


def get_config():
    c = log_config()
    c.phase.initial_lifting.kind = 'latent_additive'
    c.phase.initial_lifting.time_scale = .01
    c.boundary = ConfigDict(dict(dirichlet='exact_mouth'))
    c.arch.zero_output = True
    c.loss.gradnorm_bias = (1., 1.)
    return c
