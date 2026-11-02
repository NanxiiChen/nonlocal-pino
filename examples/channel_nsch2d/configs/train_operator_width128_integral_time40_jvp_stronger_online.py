"""Matched integral-kernel derivative control of the fixed-normalization default."""
from ml_collections import ConfigDict
from examples.channel_nsch2d.configs.train_default import get_config as baseline


def get_config():
    c=baseline()
    c.nl=ConfigDict(dict(method='integral_moment',delta=.075,neighbors=128,
        degree=3,kernel='quadratic',condition_limit=1e9))
    c.derivatives=ConfigDict(dict(kind='integral_first',boundary='integral',
        velocity='unchanged_streamfunction_ad'))
    c.phase_transport.kind='integral_conservative'
    c.log.run_name='20261102-5090-integral-time40-jvp-stronger-online-width128-01'
    return c
