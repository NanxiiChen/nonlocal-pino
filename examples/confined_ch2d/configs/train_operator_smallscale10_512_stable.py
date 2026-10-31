"""Bounded Fourier inputs and a reference-free stable-part CH residual norm."""
from examples.confined_ch2d.configs.train_operator_smallscale10_512_fourier import get_config as baseline


def get_config():
    c=baseline()
    c.arch.time_max_frequency=4.
    c.operator.metric='linear_stable_resolvent'
    c.operator.metric_workers=6
    return c
