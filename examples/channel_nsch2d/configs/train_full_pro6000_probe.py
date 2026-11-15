"""Short full-grid resource control sharing the new uniform recipe settings."""
from examples.channel_nsch2d.configs.train_uniform_pro6000 import get_config as baseline


def get_config():
    c = baseline()
    c.sample.spatial.mode = 'full'
    c.sample.spatial.node_capacity = (c.sample.Nx + 1) * (c.sample.Ny + 1)
    for key in ('tile_width', 'wall_groups', 'wall_band'):
        del c.sample.spatial[key]
    c.train.iterations = 60
    return c
