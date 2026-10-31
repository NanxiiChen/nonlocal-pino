"""Wide Fourier inputs with the original temporal bandwidth; global curriculum."""
from examples.confined_ch2d.configs.train_operator_smallscale10_512_curriculum import get_config as baseline


def get_config():
    c=baseline()
    c.arch.time_max_frequency=4.
    return c
