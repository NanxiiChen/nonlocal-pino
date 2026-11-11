"""Same-card cold channel spatial comparison with 36 physical time samples."""
from ml_collections import ConfigDict

from examples.channel_nsch2d.configs.train_default import get_config as baseline


def get_config():
    c = baseline()
    c.train.iterations = 2000
    c.train.microbatch.size = 6
    c.log.checkpoint_every = 2000
    c.sample.spatial = ConfigDict(dict(mode='full', core_count=512,
        node_capacity=(c.sample.Nx+1)*(c.sample.Ny+1), pools=8, refresh_every=50, workers=2))
    return c
