"""Same early causal control, with equivalent physical IC in(phi,s) coordinates."""
from examples.particle_corrosion2d.configs.train_physical_causal_early import get_config as early_config


def get_config():
    c=early_config()
    c.loss.initial_metric='phase_liquid'
    return c
