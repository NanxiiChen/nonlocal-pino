"""Diagnostic1000-step CPU pilot; invoke with JAX_PLATFORMS=cpu explicitly."""
from examples.confined_ch2d.configs.train_operator_smallscale10_512_initial_rhs import get_config as baseline


def get_config():
    c=baseline()
    c.train.iterations=1000
    c.log.eval_every=100
    c.log.checkpoint_every=500
    return c
