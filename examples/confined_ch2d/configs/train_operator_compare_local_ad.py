"""Full-field local-AD control preserving the width256 online50k CH recipe."""
from ml_collections import ConfigDict
from examples.confined_ch2d.configs.train_operator_compare_nonlocal import get_config as baseline


def get_config():
    c = baseline()
    c.operator.spatial_discretization = 'local_ad'
    c.operator.local_ad = ConfigDict(dict(query_chunk=128,
        boundary='existing_physical_measure_positive_times',
        phase_flux_scale=1., chemical_flux_scale=1., boundary_weight=1.))
    return c
