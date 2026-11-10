"""Historical image-only B8/Adam control with four times and width 64."""
from examples.electropolish2d.configs.train_operator_online import get_config as online_config


def get_config():
    c = online_config()
    c.arch.local_input = 'zero'
    c.log.checkpoint_step = 0  # opt-in intermediate model and monitor predictions
    return c
