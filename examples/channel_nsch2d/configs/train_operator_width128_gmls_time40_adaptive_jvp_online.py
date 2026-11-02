"""Matched adaptivecausal control: exact timeJVP and fresh continuous times."""
from ml_collections import ConfigDict
from examples.channel_nsch2d.configs.train_operator_width128_gmls_time40_adaptive_causal_online import get_config as baseline


def get_config():
    c=baseline()
    del c.time_discretization
    c.time_derivative=ConfigDict(dict(kind='jvp',sampling='continuous_stratified'))
    return c
