"""Original output and uniform times, repository AC/CH objective staggering."""
from examples.particle_corrosion2d.configs.train_physical import get_config as physical_config


def get_config():
    c=physical_config()
    c.train.iterations=5000
    c.loss.stagger_steps=2000
    c.loss.stagger_switch_every=25
    return c
