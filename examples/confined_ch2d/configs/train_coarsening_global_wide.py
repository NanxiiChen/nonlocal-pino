"""Capacity control only: same global strong PDE, hidden width128."""
from examples.confined_ch2d.configs.train_coarsening_global_strong import get_config as baseline


def get_config():
    c=baseline();c.arch.hidden_dim=128
    return c
