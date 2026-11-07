"""Conventional IC/BC warmup, squared PDEs and stronger adaptive causal focus."""
from examples.particle_corrosion2d.configs.train_physical_causal_early_ic_warmup import get_config as warmup_config


def get_config():
    c=warmup_config()
    c.train.iterations=10000
    c.loss.pde_reduction='mean_square'
    c.causal.min_mean_weight=.1
    return c
