"""Current ceramic recipe with explicit three-method pairing and1k saves."""
from examples.particle_corrosion2d.configs.train_physical_spaced_w256_method_nonlocal import get_config as baseline


def get_config():
    c = baseline()
    c.operator.method_comparison = 'rolling_fem_v1'
    c.log.checkpoint_every = 1000
    return c
