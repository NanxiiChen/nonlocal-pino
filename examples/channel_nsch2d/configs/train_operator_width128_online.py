"""Matching cold online-geometry width128 control, after single-case acceptance."""
from examples.channel_nsch2d.configs.train_operator_time_refined_online import get_config as baseline


def get_config():
    c=baseline();c.arch.width=128;c.arch.spatial_features=64
    c.log.eval_every=500
    return c
