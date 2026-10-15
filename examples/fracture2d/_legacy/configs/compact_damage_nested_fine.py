"""One further continuous scale on the original-gate-passing neural field."""
from examples.fracture2d._legacy.configs.compact_damage_nested import get_config as base


def get_config():
    c = base()
    parent = '/root/autodl-tmp/tf-logs/fracture2d/compact_damage_nested_fixedH_20261011/residual_l4'
    c.compact.output_dir = '/root/autodl-tmp/tf-logs/fracture2d/compact_damage_nested_fine_fixedH_20261011'
    c.compact.residual_source = parent
    c.compact.inherited_source = parent
    c.compact.minimum_radius = .0003
    c.compact.maximum_radius = .005
    c.compact.radius_factor = 1.05
    return c
