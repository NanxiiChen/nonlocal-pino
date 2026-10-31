"""Continue the paired25k width192 model with a continuously refreshed input pool."""
from ml_collections import ConfigDict
from examples.confined_ch2d.configs.train_operator_smallscale10_1024_width192_fp32_50k import get_config as baseline


def get_config():
    c=baseline()
    parent='/root/autodl-tmp/tf-logs/confined_ch2d/20261031-confined-geometry-b16-t16-width192-bank1024-fp32-50k-01'
    c.train.continuation=ConfigDict(dict(parent_run=parent,checkpoint=parent+'/checkpoint_025000.json'))
    c.operator.online=ConfigDict(dict(workers=6,refresh_size=32,refresh_every=100,
        prefetch_batches=2,max_points=1536,max_neighbors=128,origin_step=25000))
    return c
