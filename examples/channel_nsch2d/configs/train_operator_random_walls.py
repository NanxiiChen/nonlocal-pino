"""Random smooth stationary-wall pilot; preserves validated NS--CH training scheme."""
from ml_collections import ConfigDict
from examples.channel_nsch2d.configs.train_operator_width128_integral_time40_jvp_stronger_online import get_config as baseline


def get_config():
    c=baseline()
    c.geometry=ConfigDict(dict(kind='random_spline',length=4.,height=1.,controls=8,
        start=1.15,end=2.75,taper_width=.35,coefficient_min=.01,coefficient_max=.45,
        gap_min=.55,gap_max=.70,slope_max=1.25,curvature_max=16.,max_attempts=10000,
        coefficients=(.03,.0525,.15,.24,.225,.1125,.0525,.03,
                      .03,.0525,.12,.18,.1875,.12,.0525,.03)))
    c.operator.validation_manifest='/root/autodl-tmp/data/channel_nsch2d/random-walls-20261102-02/unseen3/manifest.json'
    c.reference.root='/root/autodl-tmp/data/channel_nsch2d/random-walls-20261102-02/development'
    c.reference.coarse_grid=(64,24)
    c.reference.fine_grid=(96,36)
    c.log.run_name=''
    return c
