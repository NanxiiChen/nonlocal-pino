"""Original output and uniform times with adaptive per-time causal weighting."""
from ml_collections import ConfigDict
from examples.particle_corrosion2d.configs.train_physical import get_config as physical_config


def get_config():
    c=physical_config()
    c.train.iterations=5000
    c.loss.use_causal=True
    c.causal=ConfigDict(dict(eps_init=1.,max_eps=100.,step_size=1.1,
        max_last_weight=.9,min_mean_weight=.5))
    return c
