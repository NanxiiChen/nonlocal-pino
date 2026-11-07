"""Original global network with equivalent Simpson-integrated AC/CH physics."""
from examples.particle_corrosion2d.configs.train_physical_causal_early_ic_warmup import get_config as warmup_config


def get_config():
    c=warmup_config()
    c.loss.use_causal=False
    del c.causal
    c.loss.time_form='integral_simpson'
    c.loss.pde_reduction='mean_square'
    c.sample.temporal.kind='integral_initial_layer'
    return c
