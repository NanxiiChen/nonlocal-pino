"""Matched 20k temporal control: joint(x,t) attention without decoder FiLM."""
from examples.particle_corrosion2d.configs.train_time_film import get_config as baseline


def get_config():
    c = baseline()
    c.arch.temporal_mode = 'joint_query'
    return c
