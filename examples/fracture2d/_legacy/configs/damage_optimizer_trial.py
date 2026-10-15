"""Same own-H neural checkpoint; explicit bounded optimizer comparisons."""
from ml_collections import ConfigDict
from examples.fracture2d._legacy.configs.local_damage_pinn_smoke import get_config as base


def get_config():
    c = base()
    c.optimization = ConfigDict()
    c.optimization.source = '/root/autodl-tmp/tf-logs/fracture2d/local_damage_pinn_smoke4_20261011_pinn/state_0001'
    c.optimization.source_field_sha256 = '38c9e1fa55c545b1b3d992ef1d61c1d23d6965e1244242d67cf0fc1f13cb8ad0'
    c.optimization.source_network_sha256 = '9c0b3d016e6eeee366e209df84a00b25585ce61f601f66b9261f1b8b81e38a5f'
    c.optimization.output = '/root/autodl-tmp/tf-logs/fracture2d/local_damage_optimizer_20261011'
    c.optimization.adam_updates = 2000
    c.optimization.adam_check_every = 200
    c.optimization.methods = ['adam_residual', 'lbfgs_residual', 'lbfgs_energy']
    c.optimization.lbfgs = ConfigDict()
    c.optimization.lbfgs.maxiter = 512
    c.optimization.lbfgs.check_every = 16
    c.optimization.lbfgs.memory = 20
    c.optimization.lbfgs.max_linesearch = 30
    c.optimization.lbfgs.gtol = 1e-12
    c.optimization.lbfgs.ftol = 0.
    c.optimization.lbfgs.residual_limit = 5e-4
    c.optimization.lbfgs.range_tolerance = c.reporting.range_atol
    return c
