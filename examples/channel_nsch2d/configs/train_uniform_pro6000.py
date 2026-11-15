"""Research uniform rolling core/halo recipe; no SOTA-default promotion."""
from examples.channel_nsch2d.configs.train_stratified_spatial import get_config as baseline


def get_config():
    c = baseline()
    c.train.iterations = 30000
    c.train.microbatch.size = 18
    c.train.lr_init = 1e-3
    c.train.lr_min = 5e-6
    c.train.lr_decay_every = 2000
    c.train.lr_decay_rate = (c.train.lr_min / c.train.lr_init) ** (
        c.train.lr_decay_every / c.train.iterations)
    c.sample.spatial.core_count = 1024
    c.sample.spatial.node_capacity = 12288
    c.sample.spatial.wall_groups = 0
    c.log.checkpoint_every = 1000
    c.log.run_name = ''
    return c
