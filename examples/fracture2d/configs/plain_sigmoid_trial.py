"""No-reference diagnostic on completed53 sweep77, testing failed own H78."""
from ml_collections import ConfigDict
from examples.fracture2d.configs.output_damage_safe_continue import get_config as base


def get_config():
    c = base()
    c.continuation.no_reference = ConfigDict()
    s = c.continuation.no_reference
    s.kind = 'pending_diagnostic'
    s.source = '/root/autodl-tmp/tf-logs/fracture2d/output_damage_sigmoid_safe_full_20261012/state_0053'
    s.completed_sweep = 77
    s.field_sha256 = 'f546133aac8cf705eaccdabeded89f8af961abd16b8f768ae44fe0c6d1997788'
    s.network_sha256 = '056691e5ff3c240a89d32ca245cb7d87312cc887d9b9caacdabd1652a856fc1e'
    s.output_dir = '/root/autodl-tmp/tf-logs/fracture2d/plain_sigmoid_fixedH_20261012'
    s.transfer_policy = 'own_neural_logit_fit'
    s.parameterization = 'inherited_reference_scales'
    s.methods = list(c.continuation.methods)
    s.lbfgs = ConfigDict(c.continuation.lbfgs.to_dict())
    return c
