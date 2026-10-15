"""Five matched-initial-field output-head controls on rejected33 own H."""
from ml_collections import ConfigDict
from examples.fracture2d._legacy.configs.compact_damage_loading_l32 import get_config as base


def get_config():
    c = base()
    c.head_trial = ConfigDict()
    c.head_trial.source = '/root/autodl-tmp/tf-logs/fracture2d/compact_damage_loading_l32_20261011/state_0033'
    c.head_trial.source_field_sha256 = 'b2f3f557383a906b641b59b9daa26467e13d99f639f936a4c8c822ed05aa45eb'
    c.head_trial.source_network_sha256 = 'cd3e3189d87cb7b784579eaa0bb3ed83c30398261e013fc92b2fa853b5db5b39'
    c.head_trial.output_dir = '/root/autodl-tmp/tf-logs/fracture2d/output_damage_fixedH_20261011'
    c.head_trial.modes = ['nonnegative', 'sigmoid', 'exp_square', 'lifted_sigmoid', 'lifted_exp_square']
    c.head_trial.methods = ['residual_l32', 'residual_l16']
    c.head_trial.initialization_epsilon = 1e-10
    c.head_trial.lbfgs = ConfigDict(c.loading.lbfgs.to_dict())
    c.head_trial.lbfgs.maxiter = 2048
    c.head_trial.lbfgs.check_every = 32
    return c
