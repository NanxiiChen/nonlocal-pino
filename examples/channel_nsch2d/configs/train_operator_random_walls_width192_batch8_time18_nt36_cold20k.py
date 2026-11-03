"""Cold20k: user-selected36random times in two equal18-time chunks."""
from examples.channel_nsch2d.configs.train_operator_random_walls_width192_batch8_time20_cold20k import get_config as baseline


def get_config():
    c = baseline()
    c.sample.Nt = 36
    c.train.microbatch.size = 18
    return c
