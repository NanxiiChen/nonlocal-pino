"""Cold FP32 capacity control with 192-wide features and preserved SOAP coverage."""
from examples.confined_ch2d.configs.train_operator_smallscale10_512_width128_fp32 import get_config as baseline


def get_config():
    c = baseline()
    c.arch.width = 192
    c.arch.spatial_features = 96
    # Four heads now have 48 features each; each block's FFN has width384.
    # Preserve preconditioning of its two axes (the old 256 cap would skip one).
    c.train.soap_max_precond_dim = 512
    return c
