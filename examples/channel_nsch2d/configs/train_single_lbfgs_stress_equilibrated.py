"""Same mixed-stress finisher, with a verified physical-field-preserving head fit."""
from ml_collections import ConfigDict
from examples.channel_nsch2d.configs.train_single_lbfgs_stress import get_config as baseline


def get_config():
    c=baseline()
    root='/root/autodl-tmp/data/channel_nsch2d/stress-head-equilibration-20261101'
    c.train.refinement.stress_initializer=ConfigDict(dict(kind='physics_auxiliary_block',
        model=root+'/model_projected.eqx',report=root+'/fit.json',quadratic=root+'/quadratic.npz'))
    return c
