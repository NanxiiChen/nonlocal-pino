"""Independent AD queries with complete, separately enforced boundary conditions."""
from ml_collections import ConfigDict
from examples.channel_nsch2d.configs.train_operator_local_ad import get_config as baseline


def get_config():
    c = baseline()
    c.sample.ad_sampling = ConfigDict(dict(kind='uniform_interior', core_count=1024, replacement=False))
    c.train.microbatch.size = 18
    return c
