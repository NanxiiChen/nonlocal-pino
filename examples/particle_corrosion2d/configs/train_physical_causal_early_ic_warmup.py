"""Known IC/BC pretraining before the same early causal original-output solve."""
from examples.particle_corrosion2d.configs.train_physical_causal_early_ic import get_config as ic_config


def get_config():
    c=ic_config()
    c.loss.data_warmup_steps=1000
    return c
