"""Bounded 2000-update production cold start with verified graph monitoring."""

from examples.icemelting3d.configs.train_base import get_config as base_config


def get_config():
    c = base_config()
    c.train.iterations = 2000
    c.train.lr_init = 1e-3
    c.sample.K_core = 256
    c.sample.Nt = 16
    c.eval.graph_reference_path = "reports/icemelting-weak-prior-20261024/reference.npz"
    return c
