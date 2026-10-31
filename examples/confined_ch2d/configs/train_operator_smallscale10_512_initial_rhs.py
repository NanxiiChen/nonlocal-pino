"""Optional prescribed initial PDE response in the generic function encoder."""
from examples.confined_ch2d.configs.train_operator_smallscale10_512_batch import get_config as baseline


def get_config():
    c=baseline()
    c.operator.initial_rhs_condition=True
    c.arch.input_dims=(5,4)
    return c
