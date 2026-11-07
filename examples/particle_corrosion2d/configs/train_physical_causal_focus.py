"""Original fields/MSE; tighten the existing adaptive causal focus thresholds."""
from examples.particle_corrosion2d.configs.train_physical_causal_early_ic_warmup_l2 import get_config as warmup_config


def get_config():
    c=warmup_config()
    c.causal.max_last_weight=.1
    c.causal.min_mean_weight=.05
    return c
