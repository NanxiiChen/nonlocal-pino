"""Current-card nonlocal full-grid timing; original cold20k LR and architecture."""
from ml_collections import ConfigDict
from examples.channel_nsch2d.configs.train_operator_random_walls_width192_batch8_time18_nt36_cold20k import get_config as baseline


def get_config():
    c = baseline()
    c.train.iterations = 60
    c.sample.spatial = ConfigDict(dict(mode='full', core_count=512,
        node_capacity=(c.sample.Nx+1)*(c.sample.Ny+1), pools=8, refresh_every=50, workers=2))
    return c
