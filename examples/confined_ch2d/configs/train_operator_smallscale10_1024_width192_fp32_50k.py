"""Cold width192/FP32/50k control with twice as many input-only geometries."""
from examples.confined_ch2d.configs.train_operator_smallscale10_512_width192_fp32 import get_config as baseline


def get_config():
    c = baseline()
    c.operator.train_conditions = 1024
    c.operator.dataset_path = '/root/autodl-tmp/data/confined_ch2d/operator_geometry_smallscale10_1024_v1'
    c.operator.precompute.source_path = '/root/autodl-tmp/data/confined_ch2d/operator_geometry_smallscale10_512_v1'
    c.operator.precompute.reuse_mode = 'hardlink'
    c.train.iterations = 50000
    return c
