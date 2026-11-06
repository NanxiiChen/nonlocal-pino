"""Cold physics-only training with rolling geometry and core/halo inputs."""
from ml_collections import ConfigDict

from examples.particle_corrosion2d.configs.train_wide import get_config as wide_config


def get_config():
    c = wide_config()
    c.operator.online = ConfigDict(dict(
        workers=4, geometry_every=250, geometry_size=4,
        spatial_every=50, spatial_size=4, prefetch=4,
        halo=3072, neighbors=64, worker_cache=64))
    return c
