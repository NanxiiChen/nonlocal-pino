"""Same B8/Nt40 objective, two gradient blocks on the qualified96GB GPU."""
from examples.channel_nsch2d.configs.train_operator_random_walls_width192_batch8 import get_config as baseline


def get_config():
    c = baseline()
    c.train.microbatch.size = 20
    return c
