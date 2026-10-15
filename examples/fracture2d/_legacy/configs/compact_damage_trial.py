"""Stage1 continuation: compact hidden units on the trained own-H trunk."""
from ml_collections import ConfigDict
from examples.fracture2d._legacy.configs.conditioned_damage_trial import get_config as base


def get_config():
    c = base()
    c.compact = ConfigDict()
    c.compact.source = '/root/autodl-tmp/tf-logs/fracture2d/conditioned_damage_fixedH_20261011/residual_l4'
    c.compact.residual_source = None
    c.compact.baseline_mode = 'raw_conditioned'
    c.compact.inherited_source = None
    c.compact.methods = ['residual_mse', 'residual_l4']
    c.compact.output_dir = '/root/autodl-tmp/tf-logs/fracture2d/compact_damage_fixedH_20261011'
    c.compact.grid_step = .02
    c.compact.coarse_radius = .035
    c.compact.adaptive_centers = 2048
    c.compact.minimum_separation = .001
    c.compact.minimum_radius = .002
    c.compact.maximum_radius = .008
    c.compact.radius_factor = 2.
    return c
