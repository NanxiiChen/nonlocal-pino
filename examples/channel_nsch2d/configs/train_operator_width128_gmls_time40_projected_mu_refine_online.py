"""Global-time physics-only model refinement with exact interior chemical recovery."""
from examples.channel_nsch2d.configs.train_operator_width128_gmls_time40_adaptive_jvp_online import get_config as baseline


def get_config():
    c=baseline()
    c.chemical_constraint=dict(kind='interior_gmls_projection',boundary='decoder_trace',query='mapped_bilinear')
    c.train.initialization=dict(kind='physics_model',optimizer='fresh',source_run=
        '/root/autodl-tmp/tf-logs/channel_nsch2d/20261102-5090-gmls-time40-adaptive-jvp-online-width128-01')
    c.train.lr_init=1e-4
    c.log.run_name='20261102-5090-gmls-time40-projected-mu-refine-online-width128-01'
    return c
