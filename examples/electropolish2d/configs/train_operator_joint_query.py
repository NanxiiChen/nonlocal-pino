"""Joint(x,t)attention without FiLM; paired with the full-test nonlocal10k arm."""
from examples.electropolish2d.configs.train_operator_threeway import get_config as baseline


def get_config():
    c = baseline()
    c.arch.temporal_query = 'joint_query'
    return c
