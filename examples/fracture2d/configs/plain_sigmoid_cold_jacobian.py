"""Same fresh.005 test with initial AT2 readout-column Jacobian scaling."""
from examples.fracture2d.configs.plain_sigmoid_cold import get_config as base


def get_config():
    c = base()
    c.cold.parameterization = 'initial_AT2_readout_jacobian_unit_trunk'
    c.cold.output_dir = '/root/autodl-tmp/tf-logs/fracture2d/plain_sigmoid_cold_d005_jacobian_20261012'
    return c
