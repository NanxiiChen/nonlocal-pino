"""Research control: same lifted128cores, half-horizon sampling wall band."""
from examples.confined_ch2d.configs.train_operator_initial_rate_spatial import get_config as baseline


def get_config():
    c=baseline()
    c.sample.spatial.wall_band=.5*c.nl.delta
    return c
