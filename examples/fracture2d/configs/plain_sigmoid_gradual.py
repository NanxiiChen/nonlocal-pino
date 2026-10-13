"""Fresh no-reference sigmoid, accepted-only uniform loading from physical zero."""
from examples.fracture2d.configs.plain_sigmoid_cold import get_config as base


def get_config():
    c = base()
    c.cold.output_dir = '/root/autodl-tmp/tf-logs/fracture2d/plain_sigmoid_gradual_20261012'
    c.cold.indices = list(range(1,76))
    c.cold.initial_damage = 1e-5
    c.cold.reference_load = .0001
    c.cold.history_feature_scaling = 'physical'
    c.cold.parameterization = 'unscaled_logits_initial_AT2_jacobian'
    c.cold.trunk_gain = 1.
    return c
