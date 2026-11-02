"""Untrained cold online counterpart of the mixed-Cauchy-stress NS control."""
from ml_collections import ConfigDict
from examples.channel_nsch2d.configs.train_operator_time_refined_online import get_config as baseline


def get_config():
    c=baseline()
    c.stress_formulation=ConfigDict(dict(kind='cauchy_mixed',initialization='preserve_original',
        constitutive_scale=c.loss.pressure_scale))
    return c
