"""Cold5k chemical-row weighting control, with unchanged physical mu amplitude."""
from examples.channel_nsch2d.configs.train_operator_width128_gmls_time40_jvp_uniform_online import get_config as baseline


def get_config():
    c=baseline()
    c.loss.chemical_residual_scale=1.
    c.log.run_name='20261102-5090-gmls-time40-jvp-chemical-online-width128-01'
    return c
