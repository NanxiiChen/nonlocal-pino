"""Same effective B8/Nt40 capacity trial, with larger time blocks for96GB GPU."""
from examples.channel_nsch2d.configs.train_operator_random_walls_width192_batch8 import get_config as baseline


def get_config():
    c = baseline()
    c.train.microbatch.size = 8
    return c
