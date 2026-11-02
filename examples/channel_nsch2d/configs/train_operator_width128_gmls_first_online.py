"""Matched cold20k residual-first-derivative control; physical outputs unchanged."""
from ml_collections import ConfigDict
from examples.channel_nsch2d.configs.train_operator_width128_dual_time_online import get_config as baseline


def get_config():
    c=baseline()
    c.derivatives=ConfigDict(dict(kind='gmls_first',boundary='gmls',
        velocity='unchanged_streamfunction_ad'))
    c.phase_transport=ConfigDict(dict(kind='gmls_conservative',quadrature='sbp4_mapped'))
    return c
