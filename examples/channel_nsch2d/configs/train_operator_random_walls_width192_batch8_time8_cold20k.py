"""Cold20k memory correction: same40-time objective in five chunks of8."""
from examples.channel_nsch2d.configs.train_operator_random_walls_width192_batch8_time20_cold20k import get_config as baseline


def get_config():
    c = baseline()
    c.train.microbatch.size = 8
    return c
