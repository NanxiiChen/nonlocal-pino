"""Single-geometry BDF2 with conservative mapped SBP phase transport."""
from ml_collections import ConfigDict
from examples.channel_nsch2d.configs.train_single_bdf2 import get_config as baseline


def get_config():
    c=baseline()
    c.phase_transport=ConfigDict(dict(kind='sbp4_mapped_curl'))
    return c
