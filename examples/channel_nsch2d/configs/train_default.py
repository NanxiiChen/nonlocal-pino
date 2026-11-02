"""Validated random-wall operator; qualified for a 96GB GPU (see README)."""
from examples.channel_nsch2d.configs.train_operator_random_walls_width192_batch8_time20 import get_config as fixed_config


def get_config():
    c=fixed_config()
    # Fresh training gets the canonical timestamp name, never an existing run.
    c.log.run_name=''
    return c
