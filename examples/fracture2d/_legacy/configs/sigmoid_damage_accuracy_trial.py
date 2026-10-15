"""Matched own-H sigmoid controls and extended critical-load53 continuation."""
from ml_collections import ConfigDict
from examples.fracture2d._legacy.configs.bounded_damage_loading import get_config as base


def get_config():
    c = base()
    c.accuracy = ConfigDict()
    a = c.accuracy
    a.source_root = '/root/autodl-tmp/tf-logs/fracture2d/bounded_damage_loading_20261013'
    a.audit_relative_path = 'comparison_current_sigmoid_0058_20261013/completed_prefix_audit.json'
    a.pins = ConfigDict({
        'identity.json':'dfaad4b0b1bc0065b07aa9a957b3f4524f88780de2051c13e945c78e1908eb09',
        'sigmoid/state_0053/fields.npz':'246e11bee16c80ab57622fc5a9c85f4b43c9f0bcc705df127641b38eb689d483',
        'sigmoid/state_0053/network.eqx':'4cbee6efb3fd460562d04b6228f6d6e04916558f8d1d0c10ef66f4f842d99c43',
        'sigmoid/state_0053/result.json':'d99a7ea143efbfcbd6def056c21a169dae7764139d005245545ec00b139ded7b',
        'fixed_0052/sigmoid/parameterization.npz':'11b29d7021c364a809dc3b9958114e04fc4f58db73c85c0d32ec84ebcd69466e',
        a.audit_relative_path:'a075a868ba23fb93154aa2f4b39d1c0cedcb56ec5a5ae0bb44ab94d71ed96709',
    }, allow_dotted_keys=True)
    a.output_dir = '/root/autodl-tmp/tf-logs/fracture2d/sigmoid_damage_accuracy_20261013'
    a.index = 53
    a.source_sweep = 128
    a.full_short = 512
    a.full_long = 8192
    a.adam_rates = [1e-4,1e-5]
    a.adam_iterations = 8192
    a.polish_iterations = 4096
    a.inner_iterations = 4096
    a.residual_limit = .001
    a.core_count = 2048
    a.halo_capacity = 32768
    a.resample_every = 5
    a.check_every = 128
    a.outer_max = 1024
    a.checkpoints = [160,256,512,768,1024]
    return c
