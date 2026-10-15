"""Certified52 own-H parameter conversion before no-reference continuation."""
from examples.fracture2d._legacy.configs.plain_sigmoid_trial import get_config as base


def get_config():
    c = base()
    s = c.continuation.no_reference
    s.kind = 'accepted_parent'
    s.source = c.continuation.parent
    s.completed_sweep = 0
    s.field_sha256 = c.continuation.parent_field_sha256
    s.network_sha256 = c.continuation.parent_network_sha256
    s.output_dir = '/root/autodl-tmp/tf-logs/fracture2d/plain_sigmoid_parent52_20261012'
    return c
