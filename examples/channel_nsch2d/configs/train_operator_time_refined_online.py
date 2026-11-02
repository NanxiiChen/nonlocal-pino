"""Untrained online counterpart of the explicit BDF2 dt.025 control."""
from examples.channel_nsch2d.configs.train_operator_conservative_online import get_config as baseline


def get_config():
    c=baseline();c.time_discretization.dt=.025
    c.operator.validation_manifest='/root/autodl-tmp/data/channel_nsch2d/unseen-geometry16-offgrid-20261101/manifest.json'
    return c
