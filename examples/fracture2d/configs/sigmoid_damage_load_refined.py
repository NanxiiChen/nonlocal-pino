"""Four quarter-load increments from the unchanged physical and sigmoid52 parents."""
from ml_collections import ConfigDict
from examples.fracture2d.configs.bounded_damage_loading import get_config as base


def get_config():
    c = base()
    c.refinement = ConfigDict()
    a = c.refinement
    a.source_root = '/root/autodl-tmp/tf-logs/fracture2d/bounded_damage_loading_20261013'
    a.audit_relative_path = 'comparison_current_sigmoid_0058_20261013/completed_prefix_audit.json'
    a.pins = ConfigDict({
        'identity.json':'dfaad4b0b1bc0065b07aa9a957b3f4524f88780de2051c13e945c78e1908eb09',
        'fixed_0052/sigmoid/network.eqx':'2ef00c129356cf261f865ac8b48085f57535e35270051aadfa0e5c11134cf8e3',
        'fixed_0052/sigmoid/fields.npz':'c4448818d266b4dd00b2e934f043c9066acb5e44a43bf0a5721c8e5843d93cd0',
        'fixed_0052/sigmoid/result.json':'0382a4eeaa47f127c83cd6355a66593ed7d8ec64b3673af4f9ae61006372fd3c',
        'fixed_0052/sigmoid/parameterization.npz':'11b29d7021c364a809dc3b9958114e04fc4f58db73c85c0d32ec84ebcd69466e',
        a.audit_relative_path:'a075a868ba23fb93154aa2f4b39d1c0cedcb56ec5a5ae0bb44ab94d71ed96709',
    },allow_dotted_keys=True)
    a.output_dir = '/root/autodl-tmp/tf-logs/fracture2d/sigmoid_damage_load_refined_20261013'
    a.parent_index = 52
    a.head = 'sigmoid'
    a.load_step = 2.5e-5
    a.load_ids = [209,210,211,212]
    a.end_load = .0053
    a.inner_iterations = 4096
    a.outer_iterations = 1024
    a.check_every = 128
    a.checkpoints = [1,8,32,128,256,512,768,1024]
    a.user_provisional = True
    return c
