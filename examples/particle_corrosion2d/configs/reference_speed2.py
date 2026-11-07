"""Evaluation-only local reference for the twice-speed experiment."""
from examples.particle_corrosion2d.configs.train_speed2 import get_config as training_config


def get_config():
    c = training_config()
    c.reference.rtol = 2e-8
    c.reference.atol = 2e-10
    c.reference.max_step = 0.00625
    return c
