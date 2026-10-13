"""Declared sigmoid continuation from accepted34 through original load75."""
from ml_collections import ConfigDict
from examples.fracture2d.configs.output_damage_loading import get_config as base


def get_config():
    c = base()
    c.continuation = ConfigDict()
    c.continuation.parent = '/root/autodl-tmp/tf-logs/fracture2d/output_damage_loading_v2_20261011/sigmoid/state_0034'
    c.continuation.parent_field_sha256 = '2cfc1404b0e8481f6494328c958fbc4326314ef6c84c641c5787343ddfb9a316'
    c.continuation.parent_network_sha256 = 'f14a1522705d5f55c786636e488f6fa5f11d78dd959e583c362886e162f11ddf'
    c.continuation.output_dir = '/root/autodl-tmp/tf-logs/fracture2d/output_damage_sigmoid_full_20261011'
    c.continuation.mode = 'sigmoid'
    c.continuation.first_index = 35
    c.continuation.last_index = 75
    c.continuation.methods = ['residual_l32', 'residual_l16']
    c.continuation.lbfgs = ConfigDict(c.head_loading.lbfgs.to_dict())
    return c
