"""Two certified bounded heads at33/34, from accepted own-neural physical32."""
from ml_collections import ConfigDict
from examples.fracture2d._legacy.configs.output_damage_trial import get_config as base


def get_config():
    c = base()
    c.head_loading = ConfigDict()
    c.head_loading.output_dir = '/root/autodl-tmp/tf-logs/fracture2d/output_damage_loading_v2_20261011'
    c.head_loading.fixed_H_root = c.head_trial.output_dir
    c.head_loading.parent = '/root/autodl-tmp/tf-logs/fracture2d/compact_damage_loading_l32_20261011/state_0032'
    c.head_loading.parent_field_sha256 = '50048516e46f288833234ba683756e0a15feb3e7bad6f9110c301bcaa16e1b01'
    c.head_loading.parent_network_sha256 = '2915448be4d1d5baf1d7e770a27c9daa035a58d602f4d60964528d183590b6d2'
    c.head_loading.modes = ['sigmoid', 'exp_square']
    c.head_loading.indices = [33, 34]
    c.head_loading.methods = ['residual_l32', 'residual_l16']
    c.head_loading.first_H_relative_limit = 1e-6
    c.head_loading.lbfgs = ConfigDict(c.head_trial.lbfgs.to_dict())
    return c
