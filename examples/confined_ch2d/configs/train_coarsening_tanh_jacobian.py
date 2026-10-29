"""Same phase-only tanh map; divide the physical PDE residual by its Jacobian."""
from examples.confined_ch2d.configs.train_coarsening_tanh import get_config as baseline


def get_config():
    c=baseline();c.loss.residual_transform.kind='inverse_tanh_jacobian'
    return c
