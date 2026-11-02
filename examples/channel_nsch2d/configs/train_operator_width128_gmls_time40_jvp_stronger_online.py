"""Moderately deeper channel walls, retaining the validated chemical-row1 recipe."""
from examples.channel_nsch2d.configs.train_operator_width128_gmls_time40_jvp_chemical_online import get_config as baseline


def get_config():
    c=baseline()
    c.geometry.upper=.18
    c.geometry.lower=.12
    c.geometry.amplitude_min=.06
    c.geometry.amplitude_max=.20
    c.reference.root='/root/autodl-tmp/data/channel_nsch2d/stronger-reference-20261102-01'
    c.operator.validation_manifest='/root/autodl-tmp/data/channel_nsch2d/stronger-unseen16-20261102-01/manifest.json'
    c.log.run_name='20261102-5090-gmls-time40-jvp-stronger-online-width128-01'
    return c
