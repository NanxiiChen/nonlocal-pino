"""One global network; adaptive causal weights, never frozen time states."""
from ml_collections import ConfigDict
from examples.confined_ch2d.configs.train_coarsening_global_strong import get_config as baseline


def get_config():
    c=baseline();c.loss.causal=ConfigDict(dict(strength=1.,minimum=1e-8,refinement='uniform'))
    return c
