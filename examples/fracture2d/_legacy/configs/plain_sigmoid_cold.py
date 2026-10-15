"""One-load applicability test: fresh plain sigmoid and own-zero state at.005."""
from ml_collections import ConfigDict
from examples.fracture2d._legacy.configs.plain_sigmoid_trial_rms import get_config as base


def get_config():
    previous = base()
    c = ConfigDict({key: previous[key].to_dict() for key in
        ('physical', 'geometry', 'nl', 'arch', 'train', 'data', 'trial',
         'conditioned', 'validation', 'outer_stop', 'reporting', 'stagger')})
    c.cold = ConfigDict()
    c.cold.output_dir = '/root/autodl-tmp/tf-logs/fracture2d/plain_sigmoid_cold_d005_20261012'
    c.cold.indices = [50]
    c.cold.initial_damage = .01
    c.cold.reference_load = .005
    c.cold.history_feature_scaling = 'load_squared'
    c.cold.basis_path = '/root/autodl-tmp/tf-logs/fracture2d/compact_damage_polish_fixedH_20261011/basis.npz'
    c.cold.basis_sha256 = '9b56ba9b607dfc861039ed5b5ce1a79c3542b819e8a4e2f7530db79e01d7b204'
    c.cold.parameterization = 'latent_basis_rms_unit_trunk_at_reference'
    c.cold.methods = ['residual_l32', 'residual_l16']
    c.cold.lbfgs = ConfigDict(previous.continuation.lbfgs.to_dict())
    return c
