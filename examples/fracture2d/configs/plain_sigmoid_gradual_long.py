"""Same fresh gradual experiment with a declared larger inner training budget."""
from examples.fracture2d.configs.plain_sigmoid_gradual import get_config as base


def get_config():
    c = base()
    c.cold.output_dir = '/root/autodl-tmp/tf-logs/fracture2d/plain_sigmoid_gradual_long_20261012'
    c.cold.lbfgs.maxiter = 8192
    return c
