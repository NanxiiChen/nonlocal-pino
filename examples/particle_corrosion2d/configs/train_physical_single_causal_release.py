"""Early adaptive causal training,then joint full-time PDE optimization."""
from examples.particle_corrosion2d.configs.train_physical_single_gn_l2 import get_config as norm_config


def get_config():
    c=norm_config()
    c.loss.causal_steps=5000
    return c
