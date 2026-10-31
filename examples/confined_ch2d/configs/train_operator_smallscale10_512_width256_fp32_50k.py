"""Cold width256 FP32 operator with a fixed 50k SOAP budget."""
from examples.confined_ch2d.configs.train_operator_smallscale10_512_width192_fp32 import get_config as baseline


def get_config():
    c = baseline()
    c.arch.width = 256
    c.arch.spatial_features = 128
    c.train.iterations = 50000
    return c
