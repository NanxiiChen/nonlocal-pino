"""Matched dense-BC cold20k control using existing inverse-gradient weighting."""
from examples.channel_nsch2d.configs.train_operator_width128_gmls_densebc_online import get_config as baseline
from examples.channel_nsch2d.configs.train_stream_gradnorm import get_config as balanced


def get_config():
    c=baseline();c.balance=balanced().balance
    return c
