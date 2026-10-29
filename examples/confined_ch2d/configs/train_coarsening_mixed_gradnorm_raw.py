"""Raw mixed residuals scaled only by rounded known-initial-field RMS values."""
from examples.confined_ch2d.configs.train_coarsening_mixed_gradnorm import get_config as baseline


def get_config():
    c=baseline()
    del c.loss.phase_metric
    c.loss.evolution_scale=100.
    return c
