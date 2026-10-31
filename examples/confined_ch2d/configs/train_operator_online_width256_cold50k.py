"""Fresh width256/FP32/50k training with a bounded online geometry pool."""
from ml_collections import ConfigDict
from examples.confined_ch2d.configs.train_operator_smallscale10_1024_width192_fp32_50k import get_config as baseline


def get_config():
    c=baseline()
    c.arch.width=256
    c.arch.spatial_features=128
    c.operator.online=ConfigDict(dict(workers=6,refresh_size=32,refresh_every=100,
        prefetch_batches=2,max_points=1536,max_neighbors=128,origin_step=0))
    # Ten paired recovery points instead of fifty; retain every saved state.
    c.log.checkpoint_every=5000
    return c
