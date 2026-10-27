"""Fixed integer periodic features; original residual with PDE-only refinement."""
from ml_collections import ConfigDict
from examples.twophase2d.configs.train_vortex import get_config as vortex_config


def get_config():
    c = vortex_config()
    c.train.compute_dtype = 'float64'
    c.train.refinement = ConfigDict(dict(optimizer='lbfgs', iterations=1000,
        max_evaluations=2000, max_line_search=40, history_size=50, time_points=32,
        gradient_tolerance=1e-9, function_tolerance=1e-14))
    return c
