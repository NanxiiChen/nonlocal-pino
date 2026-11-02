"""Untrained384x160online counterpart; explicit B1/Nt4memory budget."""
from examples.channel_nsch2d.configs.train_operator_time_refined_online import get_config as baseline


def get_config():
    c=baseline();c.sample.Nx=384;c.sample.Ny=160;c.operator.condition_batch=1
    return c
