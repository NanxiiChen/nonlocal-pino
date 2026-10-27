"""Long-refinement budget control; not recommended because field L2 can worsen."""
from examples.twophase2d.configs.train_vortex_conserved import get_config as conserved_config


def get_config():
    c = conserved_config()
    c.phase.map = 'inverse_cubic'
    c.train.refinement.iterations = 3000
    c.train.refinement.max_evaluations = 5000
    return c
