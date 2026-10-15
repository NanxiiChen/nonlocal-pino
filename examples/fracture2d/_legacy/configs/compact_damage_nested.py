"""Preserve the trained multiscale basis and add residual-selected finer units."""
from examples.fracture2d._legacy.configs.compact_damage_multiscale import get_config as base


def get_config():
    c = base()
    parent = '/root/autodl-tmp/tf-logs/fracture2d/compact_damage_multiscale_fixedH_20261011/residual_l4'
    c.compact.output_dir = '/root/autodl-tmp/tf-logs/fracture2d/compact_damage_nested_fixedH_20261011'
    c.compact.residual_source = parent
    c.compact.inherited_source = parent
    c.compact.adaptive_centers = 4096
    c.compact.minimum_radius = .0004
    c.compact.maximum_radius = .006
    c.compact.radius_factor = 1.2
    return c
