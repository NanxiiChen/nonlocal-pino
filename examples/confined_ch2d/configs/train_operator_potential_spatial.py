"""Experimental two-hop rolling cores; retain the full geometry encoder."""
from ml_collections import ConfigDict
from examples.confined_ch2d.configs.train_operator_potential import get_config as baseline


def get_config():
    c=baseline()
    c.sample.spatial=ConfigDict(dict(mode='rolling_groups',core_count=128,group_size=64,
        cell_width=.12,wall_groups=1,wall_band=c.nl.delta,inner_capacity=512,
        query_capacity=1024,pools=4,refresh_every=50,workers=2))
    return c
