"""Width128 control of the selected cold30k recipe; all other settings fixed."""
from examples.confined_ch2d.configs.train_operator_smallscale10_best import get_config as baseline


def get_config():
    c = baseline()
    c.arch.width = 128
    return c
