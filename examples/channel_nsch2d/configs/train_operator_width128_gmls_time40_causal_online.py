"""Matched cold5k,40times; stopped coupled temporal prefixes across all blocks."""
from ml_collections import ConfigDict
from examples.channel_nsch2d.configs.train_operator_width128_gmls_time40_online import get_config as baseline


def get_config():
    c=baseline()
    c.causal=ConfigDict(dict(kind='coupled_prefix_mse',
        epsilon_values=(.01,.1,1.,10.,100.),steps_per_epsilon=1000,
        weight_gradient='stop',boundary='same_time_weights'))
    return c
