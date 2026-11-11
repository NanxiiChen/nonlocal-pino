"""Current rolling recipe with explicit nonlocal/local-AD research receipts."""
from examples.electropolish2d.configs.train_operator import get_config as current_config


def get_config():
    c = current_config()
    c.operator.method_control = 'rolling_derivative_pair'
    c.log.checkpoint_step = 5000  # Common local-reference comparison at 5k and 10k.
    return c
