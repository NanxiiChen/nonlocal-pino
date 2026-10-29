"""One-field CH with general time/space resolution matching; no trajectory data."""
from examples.confined_ch2d.configs.train_coarsening_eliminated import get_config as baseline


def get_config():
    c=baseline()
    c.phase.lifting='logarithmic_blend';c.phase.initial_time_scale=.01
    c.arch.spatial_modes=8
    return c
