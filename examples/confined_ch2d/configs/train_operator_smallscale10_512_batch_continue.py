"""Continue the complete10k B16/Nt16 parent to20k with its full SOAP state."""
from ml_collections import ConfigDict
from examples.confined_ch2d.configs.train_operator_smallscale10_512_batch import get_config as baseline


def get_config():
    c=baseline()
    c.train.iterations=20000
    c.train.continuation=ConfigDict(dict(
        parent_run='/root/autodl-tmp/tf-logs/confined_ch2d/20261030-confined-geometry-b16-t16-soap-01'))
    return c
