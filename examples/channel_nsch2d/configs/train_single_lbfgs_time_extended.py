"""Keep the complete learned linear-time function and add zero-weight log modes."""
from ml_collections import ConfigDict
from examples.channel_nsch2d.configs.train_single_lbfgs import get_config as baseline


def get_config():
    c=baseline();c.arch.time_features=16
    c.time_embedding=ConfigDict(dict(kind='linear_log',scale=c.lifting.time_scale,initialization='preserve_linear'))
    c.log.checkpoint_every=25
    return c
