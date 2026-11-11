"""Selected 3D melting PINO: Adam10000 with rolling complete core/halo sets."""
from examples.icemelting3d.configs.train_operator_rolling_pool import get_config as baseline


def get_config():
    c = baseline()
    c.train.iterations = 10000
    return c
