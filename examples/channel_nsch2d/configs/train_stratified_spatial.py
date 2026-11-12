"""Research-only rolling local groups with explicit wall/bulk volume strata."""
from examples.channel_nsch2d.configs.train_spatial_control import get_config as baseline


def get_config():
    c = baseline()
    c.sample.spatial.mode = 'stratified_tiles'
    c.sample.spatial.tile_width = 4
    c.sample.spatial.wall_groups = 8
    c.sample.spatial.wall_band = c.nl.delta
    # Conservative full capacity for preflight; shrink only after support audit.
    c.sample.spatial.node_capacity = (c.sample.Nx+1)*(c.sample.Ny+1)
    return c
