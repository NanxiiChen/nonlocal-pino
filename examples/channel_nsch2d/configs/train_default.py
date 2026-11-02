"""Validated stronger-channel recipe with fixed residual normalization."""
from examples.channel_nsch2d.configs.train_operator_width128_gmls_time40_jvp_stronger_online import get_config as fixed_config


def get_config():
    c=fixed_config()
    # Fresh training gets the canonical timestamp name, never an existing run.
    c.log.run_name=''
    return c
