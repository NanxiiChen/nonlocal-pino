"""Lower-noise geometry/time minibatches with the same expected PDE objective."""
from examples.confined_ch2d.configs.train_operator_smallscale10_512_unit_linear import get_config as baseline


def get_config():
    c=baseline()
    c.operator.condition_batch=16
    c.sample.Nt=16
    c.loss.initial_residual_fraction=.2  # Preserve the original4positive+0 expectation.
    c.train.iterations=10000
    return c
