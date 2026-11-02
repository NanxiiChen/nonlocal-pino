"""Nine adaptive gradient weights, raw residuals, unchanged physical outputs."""
from ml_collections import ConfigDict
from examples.channel_nsch2d.configs.train_operator_width128_gmls_time40_jvp_stronger_online import get_config as baseline


def get_config():
    c=baseline()
    c.loss.phase_scale=1.
    c.loss.boundary_mu_scale=1.
    c.balance=ConfigDict(dict(kind='gradnorm',norm='l1',grouping='terms',
        epsilon=1e-6,cap=1000.,new_weight_fraction=.9,update_every=100))
    c.log.run_name='20261102-5090-gmls-time40-jvp-allterms-online-width128-01'
    return c
