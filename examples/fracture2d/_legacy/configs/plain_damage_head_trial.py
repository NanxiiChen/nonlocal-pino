"""Same-own-H comparison of sigmoid, exp-square and unconstrained outputs."""
from ml_collections import ConfigDict
from examples.fracture2d._legacy.configs.plain_sigmoid_gradual import get_config as base


def get_config():
    c = base()
    c.head = ConfigDict()
    c.head.output_dir = '/root/autodl-tmp/tf-logs/fracture2d/plain_damage_head_trial_20261012'
    c.head.source_dir = '/root/autodl-tmp/tf-logs/fracture2d/plain_sigmoid_gradual_20261012'
    c.head.source_fields_sha256 = '767f04e75be75e1d93339a83297aa1079007506f5379a4c09704de08f3f6e13c'
    c.head.source_identity_sha256 = '9cef1689930652ddf61c3b6ce9a4b73ba6935fe7f57173da30e04d789738cc79'
    c.head.source_audit_sha256 = '9cd9c77fc6d6ea2df58cedb3cf9133ba93c60914b4a1143a2af8c9993437c6f2'
    c.head.source_zero_sha256 = '383130e5e1ae5d6a4210513a9d96055d9cd7f781a0e4d260124fee79bf844412'
    c.head.load = .0001
    c.head.initial_damage = 1e-5
    c.head.heads = ['sigmoid','exp_square','identity']
    c.head.methods = ['residual_l32','residual_l16']
    c.head.lbfgs = ConfigDict(c.cold.lbfgs.to_dict())
    return c
