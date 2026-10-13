"""Peak-focused neural readout objective with the unchanged5e-4 gate."""
from examples.fracture2d.configs.compact_damage_nested_fine import get_config as base


def get_config():
    c = base()
    parent = '/root/autodl-tmp/tf-logs/fracture2d/compact_damage_nested_fine_fixedH_20261011/residual_l4'
    c.compact.output_dir = '/root/autodl-tmp/tf-logs/fracture2d/compact_damage_peak_fixedH_20261011'
    c.compact.residual_source = parent
    c.compact.inherited_source = parent
    c.compact.adaptive_centers = 1024
    c.compact.methods = ['residual_l16']
    return c
