"""Matched uniform cold5k: physical timeJVP and fresh continuous strata."""
from ml_collections import ConfigDict
from examples.channel_nsch2d.configs.train_operator_width128_gmls_time40_online import get_config as baseline


def get_config():
    c=baseline()
    del c.time_discretization
    c.time_derivative=ConfigDict(dict(kind='jvp',sampling='continuous_stratified'))
    c.log.run_name='20261102-5090-gmls-time40-jvp-uniform-online-width128-01'
    return c
