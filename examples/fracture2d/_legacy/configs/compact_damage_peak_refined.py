"""Retain the near-target peak-trained field and refine its remaining extremes."""
from examples.fracture2d._legacy.configs.compact_damage_peak import get_config as base


def get_config():
    c = base()
    parent = '/root/autodl-tmp/tf-logs/fracture2d/compact_damage_peak_fixedH_20261011/residual_l16'
    c.compact.output_dir = '/root/autodl-tmp/tf-logs/fracture2d/compact_damage_peak_refined_fixedH_20261011'
    c.compact.residual_source = parent
    c.compact.inherited_source = parent
    c.compact.adaptive_centers = 2048
    c.compact.radius_factor = 1.01
    return c
