"""Original outputs; resolve the reaction layer using resampled PDE times."""
from ml_collections import ConfigDict
from examples.particle_corrosion2d.configs.train_physical_causal import get_config as causal_config


def get_config():
    c=causal_config()
    reaction_time=1/(2*c.physical.kappa*c.physical.n_ac2)
    c.sample.temporal=ConfigDict(dict(kind='initial_layer',early_count=8,
        early_end=4*reaction_time))
    return c
