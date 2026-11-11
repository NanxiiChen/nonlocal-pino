"""Cold 10000-update fixed/rolling pool control with saved monitor snapshots."""
from examples.icemelting3d.configs.train_operator_fixed_pool import get_config as baseline


def get_config():
    c = baseline()
    c.train.iterations = 10000
    c.log.model_every = 250
    return c
