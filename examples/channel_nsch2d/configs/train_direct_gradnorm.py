"""Full direct NS--CH including soft continuity, with matched GradNorm."""
from examples.channel_nsch2d.configs.train_stream_gradnorm import get_config as baseline


def get_config():
    c=baseline()
    del c.velocity
    return c
