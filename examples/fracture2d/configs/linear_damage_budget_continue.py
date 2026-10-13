"""Declared same-load recovery: extra training, unchanged physics/gates."""
from ml_collections import ConfigDict
from examples.fracture2d.configs.linear_damage_continue import get_config as base


def get_config():
    c = base()
    c.range.lbfgs.maxiter = 8192
    c.loading.output_dir = '/root/autodl-tmp/tf-logs/fracture2d/linear_damage_budget_continue_20261012'
    c.continuation = ConfigDict()
    c.continuation.source_dir = '/root/autodl-tmp/tf-logs/fracture2d/linear_damage_continue_20261012'
    c.continuation.accepted_parent_index = 52
    c.continuation.completed_outer = 123
    c.continuation.range_atol = 2e-6
    c.continuation.source = ConfigDict({
        'identity.json': 'd9f38a1374dfd9c656e4e4cb8b7b900727721d58bf9a6b115416cd502450f95d',
        'audit.json': 'a4b46ea77c1de9154047c45f5c3fe8af2d174b33966476236643f7358e50ba4e',
        'result.json': '00a627b36e315f685d5d054b30d407cbd52d49be4884566a47d6dd0f5d96c3db',
        'parameterization.npz': '777fe3ea578a8da719a7efd22e74685ac182bf0c82ca5bb74e012ce6763fe206',
        'accepted_parent_result.json': '0ce5f8a20309a2cfd040d3ee6a5dc5623946b43bacca9d87e822395bfe9bfe81',
        'accepted_parent_fields.npz': '94ea359d4d84d074efd6b68fe5675b06fbf52831ab852f5dfce5ffa9e7654ffb',
        'state_0053/result.json': '764ef42a6418eb3d574fd121f8acaebd2c8835350e97cdcc02befaf6e3cc61dc',
        'state_0053/fields.npz': 'd42b223d59b0c0c04f1054beadbb58fb78409258cb334f71e76c662cffdc2161',
        'state_0053/network.eqx': 'c2a984bd7fee1d3e2becfca1109ef71c95c5b1c32b6152360b0721f1a5638836',
        'state_0053/pending.json': '4b0371fdb56e410a309f7e0da630f4c76c0292d3a21a9774abb621a142c9e9c2',
        'state_0053/pending_0123_fields.npz': '092d426691aa85b7366ed6f81e6b7e2790eb9a788183b50759b239ebaf6d2de8',
        'state_0053/pending_0123_network.eqx': 'b040b03d2f6d57c5f67f7770969649c6b99dd5100dd0af0c9c0c2ec12dcbcbf1',
        'state_0053/inner_0124.json': 'b681b29dea5e37833a6c62c1fe06f503b34e28af29334e10b4d50492f6bc8eb9',
    },allow_dotted_keys=True)
    return c
