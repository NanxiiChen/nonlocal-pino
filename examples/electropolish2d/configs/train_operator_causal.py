"""Matched online B8/5000 control with existing per-time causal weighting."""
from examples.electropolish2d.configs.train_operator_online import get_config as online_config


def get_config():
    c = online_config()
    c.loss.use_causal = True
    c.causal.n_chunks = 0
    c.causal.eps_init = 1.0
    c.causal.max_eps = 100.0
    c.causal.step_size = 1.1
    c.causal.max_last_weight = .9
    c.causal.min_mean_weight = .01
    return c
