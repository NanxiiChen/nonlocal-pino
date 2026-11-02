"""Full-state temporal refinement, retaining the best audited spatial scheme."""
from ml_collections import ConfigDict
from examples.channel_nsch2d.configs.train_single_conservative_refined import get_config as baseline


def get_config():
    c=baseline();c.time_discretization.dt=.025;c.train.iterations=15000
    c.train.continuation=ConfigDict(dict(kind='time_refine',
        parent_run='/root/autodl-tmp/tf-logs/channel_nsch2d/20261101-5090-refinement-channel-single-sbp-refined-01'))
    return c
