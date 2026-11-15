"""Full nodal P1 weak-CH control, preserving both original geometry encoders."""
from ml_collections import ConfigDict
from examples.confined_ch2d.configs.train_operator_compare_nonlocal import get_config as baseline


def get_config():
    c = baseline()
    c.operator.spatial_discretization = 'fem_p1_lumped'
    c.operator.fem = ConfigDict(dict(kind='p1_lumped', max_points=1024,
        max_neighbors=32, reaction='nodal', evaluation='neural_phase_p1_chemical'))
    return c
