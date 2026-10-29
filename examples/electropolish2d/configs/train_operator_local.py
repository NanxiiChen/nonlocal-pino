"""Original rough-interface operator: local AD and explicit side-wall Neumann."""
from examples.electropolish2d.configs.train_operator_online import get_config as online_config


def get_config():
    c = online_config()
    c.arch.local_input = 'zero'
    c.nl.method = 'local_ad'
    c.loss.gradnorm_bias = tuple(c.loss.gradnorm_bias)+(1.,)
    c.loss.neumann_scale = 1.0  # MSE of (phi_n, q_n/c_le), positive times only
    return c
