"""Continue only accepted31-state neural ancestry with an L32 peak objective."""
from examples.fracture2d._legacy.configs.compact_damage_loading_peak_first import get_config as base


def get_config():
    c = base()
    c.loading.output_dir = '/root/autodl-tmp/tf-logs/fracture2d/compact_damage_loading_l32_20261011'
    c.loading.methods = ['residual_l32', 'residual_l16']
    c.loading.fork.parent_root = '/root/autodl-tmp/tf-logs/fracture2d/compact_damage_loading_peak_first_20261011'
    c.loading.fork.parent_identity_sha256 = '3bc0f92a673a4603341208ed0c6087c5ee520ea39547cb2e49da7bb3e922f1a8'
    c.loading.fork.prefix_length = 31
    return c
