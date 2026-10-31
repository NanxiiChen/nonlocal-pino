"""Cold width128/beta09 FP32 control on the unchanged geometry bank."""
from examples.confined_ch2d.configs.train_operator_smallscale10_512_width128_beta09 import get_config as baseline


def get_config():
    c = baseline()
    c.train.compute_dtype = 'float32'
    return c
