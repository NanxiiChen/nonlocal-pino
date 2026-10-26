"""Historical ordinary FP32 convergence control; canonical default uses FP64."""
from examples.corrosion2d.configs.train_base import get_config as base


def get_config():
    c=base()
    c.train.gpu_policy="ordinary"
    c.train.compute_dtype="float32"
    return c
