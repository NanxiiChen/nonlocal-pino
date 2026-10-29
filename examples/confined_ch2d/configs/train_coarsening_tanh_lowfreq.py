"""Same tanh/Jacobian model with spatial Fourier modes1..5 instead of1..8."""
from examples.confined_ch2d.configs.train_coarsening_tanh_jacobian import get_config as baseline


def get_config():
    c=baseline();c.arch.spatial_modes=5
    return c
