"""Fixed geometry library; unchanged SOTA PDE,20kFiLM and rolling spatial pool."""
from examples.particle_corrosion2d.configs.train_time_film import get_config as baseline


def get_config():
    c = baseline()
    c.operator.geometry_sampling = 'fixed_library'
    return c
