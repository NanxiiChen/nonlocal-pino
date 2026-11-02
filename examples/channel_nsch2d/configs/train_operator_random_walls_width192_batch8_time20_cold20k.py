"""Cold20k control: same validated physics, exponential LR1e-3 ->1e-5."""
from examples.channel_nsch2d.configs.train_operator_random_walls_width192_batch8_time20 import get_config as baseline


def get_config():
    c = baseline()
    c.train.iterations = 20000
    c.train.lr_decay_rate = 10.**(-.2)
    c.log.run_name = ''
    return c
