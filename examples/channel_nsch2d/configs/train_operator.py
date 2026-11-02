"""Small geometry-bank full NS--CH operator trial; no SOTA claim."""
from examples.channel_nsch2d.configs.train_base import get_config as baseline


def get_config():
    c=baseline()
    c.train.optimizer='soap'
    return c
