"""Cold10k capacity trial; unchanged random-wall physics, inputs and losses."""
from examples.channel_nsch2d.configs.train_operator_random_walls import get_config as baseline


def get_config():
    c = baseline()
    c.arch.width = 192
    c.operator.condition_batch = 8
    c.train.microbatch.size = 2
    c.train.iterations = 10000
    return c
