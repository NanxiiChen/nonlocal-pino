"""Matched cold5k with the existing bidirectional causal weight feedback."""
from ml_collections import ConfigDict
from examples.channel_nsch2d.configs.train_operator_width128_gmls_time40_online import get_config as baseline


def get_config():
    c=baseline()
    c.causal=ConfigDict(dict(kind='coupled_prefix_mse',weight_gradient='stop',
        boundary='same_time_weights',adaptation=dict(kind='weight_feedback',
            eps_init=1.,step_size=1.1,max_eps=100.,max_last_weight=.99,
            min_mean_weight=.5,geometry_reduction='minimum')))
    return c
