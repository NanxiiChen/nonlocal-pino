"""Matched current rolling local-AD control; original rough interfaces retained."""
from examples.electropolish2d.configs.train_operator_rolling_nonlocal import get_config as paired_config


def get_config():
    c = paired_config()
    c.nl.method = 'local_ad'
    c.loss.gradnorm_bias = tuple(c.loss.gradnorm_bias) + (1.,)
    c.loss.neumann_scale = 1.0  # MSE of (phi_n, q_n/c_le), at positive times only
    return c
