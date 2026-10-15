"""Own-zero two-load transfer test; all original physical gates retained."""
from ml_collections import ConfigDict
from examples.fracture2d._legacy.configs.compact_damage_polish import get_config as base


def get_config():
    c = base()
    c.loading = ConfigDict()
    c.loading.source = '/root/autodl-tmp/tf-logs/fracture2d/compact_damage_polish_fixedH_20261011/residual_l32'
    c.loading.output_dir = '/root/autodl-tmp/tf-logs/fracture2d/compact_damage_loading_20261011'
    c.loading.methods = ['residual_mse', 'residual_l16']
    c.loading.train_trunk = False
    c.loading.history_feature_scaling = 'physical'
    c.loading.lbfgs = ConfigDict(c.optimization.lbfgs.to_dict())
    c.loading.lbfgs.maxiter = 512
    c.loading.lbfgs.check_every = 16
    c.outer_stop.max_sweeps = 2048
    c.data.reference_indices = list(range(1, 76))
    c.data.end_state_index = 2
    c.validation.cg_atol = 1e-13
    return c
