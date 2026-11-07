"""Single fixed-geometry optimization diagnostic; original strong residuals."""
from examples.particle_corrosion2d.configs.train_physical_causal_focus import get_config as focus_config


def get_config():
    c=focus_config()
    c.operator.geometry_count=1
    c.operator.condition_batch=1
    c.operator.ic_points=4096
    del c.operator.online
    c.sample.K_core=1024
    c.sample.K_pool=32
    return c
