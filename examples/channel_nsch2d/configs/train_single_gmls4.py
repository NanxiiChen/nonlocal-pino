"""Full-state operator refinement of the audited192x80/global10000 model."""
from ml_collections import ConfigDict
from examples.channel_nsch2d.configs.train_single_conservative_refined import get_config as baseline


def get_config():
    c=baseline();c.nl.degree=4;c.nl.neighbors=48;c.nl.weight_exponent=12.
    c.train.iterations=15000
    c.train.continuation=ConfigDict(dict(kind='operator',
        parent_run='/root/autodl-tmp/tf-logs/channel_nsch2d/20261101-5090-refinement-channel-single-sbp-refined-01'))
    return c
