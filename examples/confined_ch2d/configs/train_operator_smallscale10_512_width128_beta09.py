"""Cold width128 SOAP beta2=.9 control; all other selected settings fixed."""
from examples.confined_ch2d.configs.train_operator_smallscale10_512_width128 import get_config as baseline


def get_config():
    c = baseline()
    c.train.soap_b2 = .9
    return c
