"""Natural-Neumann lumped P1 weak residual; unchanged physical field map."""
from ml_collections import ConfigDict
from examples.particle_corrosion2d.configs.train_fem_comparison_nonlocal import get_config as baseline


def get_config():
    c = baseline()
    c.nl.method = 'fem_weak'
    c.fem = ConfigDict(dict(kind='p1_lumped_nodal', residual_norm='mass_normalized_l1'))
    c.operator.online.halo = 2048
    c.operator.online.neighbors = 32
    return c
