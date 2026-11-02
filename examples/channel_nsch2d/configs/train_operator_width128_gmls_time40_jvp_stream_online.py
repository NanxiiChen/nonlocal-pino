"""Matched JVP control with the existing normalized inlet gate for velocity."""
from examples.channel_nsch2d.configs.train_operator_width128_gmls_time40_adaptive_jvp_online import get_config as baseline


def get_config():
    c=baseline()
    c.stream_lifting=dict(kind='normalized_inlet')
    c.log.run_name='20261102-5090-gmls-time40-jvp-stream-online-width128-01'
    return c
