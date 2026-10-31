"""Continue the complete width96/20k model to30k with unchanged SOAP schedule."""
from examples.confined_ch2d.configs.train_operator_smallscale10_512_wide_continue import get_config as baseline


def get_config():
    c=baseline()
    c.train.iterations=30000
    c.train.continuation.parent_run=(
        '/root/autodl-tmp/tf-logs/confined_ch2d/20261030-confined-geometry-b16-t16-wide96-continue-01')
    return c
