"""Mixed-Cauchy-stress full-time finisher; preserve the original physical fields."""
from ml_collections import ConfigDict
from examples.channel_nsch2d.configs.train_single_lbfgs import get_config as baseline


def get_config():
    c=baseline()
    c.stress_formulation=ConfigDict(dict(kind='cauchy_mixed',initialization='preserve_original',
        constitutive_scale=c.loss.pressure_scale))
    c.log.checkpoint_every=25
    return c
