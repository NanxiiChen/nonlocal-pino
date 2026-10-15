"""Allow fine units at coarse tip/corner centers using neural residuals."""
from examples.fracture2d._legacy.configs.compact_damage_trial import get_config as base


def get_config():
    c = base()
    c.compact.output_dir = '/root/autodl-tmp/tf-logs/fracture2d/compact_damage_multiscale_fixedH_20261011'
    c.compact.residual_source = '/root/autodl-tmp/tf-logs/fracture2d/compact_damage_fixedH_20261011/residual_l4'
    c.compact.adaptive_centers = 8192
    c.compact.minimum_separation = .0002
    c.compact.minimum_radius = .0006
    c.compact.maximum_radius = .008
    return c
