"""Joint(x,t)attention without FiLM; nonlocal, hardIC and latent PDE unchanged."""
from examples.icemelting3d.configs.train_operator_compare_nonlocal import get_config as baseline


def get_config():
    c = baseline()
    c.arch.temporal_query = 'joint_query'
    return c
