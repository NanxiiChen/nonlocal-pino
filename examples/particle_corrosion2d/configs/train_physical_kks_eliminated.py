"""Original outputs with a bounded equivalent KKS residual row elimination."""
from examples.particle_corrosion2d.configs.train_physical_causal_focus import get_config as focus_config


def get_config():
    c=focus_config()
    c.loss.transport_residual='ac_eliminated'
    return c
