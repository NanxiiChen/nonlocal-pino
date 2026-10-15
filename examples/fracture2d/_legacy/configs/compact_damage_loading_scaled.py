"""Joint control with load-squared-normalized known-H neural features only."""
from examples.fracture2d._legacy.configs.compact_damage_loading_joint import get_config as base


def get_config():
    c = base()
    c.loading.output_dir = '/root/autodl-tmp/tf-logs/fracture2d/compact_damage_loading_scaled_20261011'
    c.loading.history_feature_scaling = 'load_squared'
    return c
