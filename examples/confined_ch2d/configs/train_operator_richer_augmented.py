"""Same richer benchmark with reflection-consistent physics-only augmentation."""
from examples.confined_ch2d.configs.train_operator_richer import get_config as baseline


def get_config():
    c=baseline()
    c.operator.augmentation='axis_reflections'
    return c
