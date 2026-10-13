"""Neural predictor plus explicit AT2 nodal correction; not pure PINN."""
from ml_collections import ConfigDict
from examples.fracture2d.configs.local_damage_trial_2048 import get_config as base


def get_config():
    c = base()
    c.trial.damage_solver = 'pinn_with_nodal_correction'
    c.trial.output_base = '/root/autodl-tmp/tf-logs/fracture2d/local_damage_corrected_20261011'
    c.trial.fork_source = ''
    c.trial.fork_ready_loads = 0
    c.predictor = ConfigDict()
    c.predictor.source = '/root/autodl-tmp/tf-logs/fracture2d/local_damage_optimizer_20261011/lbfgs_energy/network.eqx'
    c.predictor.source_sha256 = '97f1b7e9c1b1f3adf3dfd29f90252ade47cdacda14adc8d3f58373983f97af88'
    c.predictor.initial_projection = 'none'
    c.predictor.lbfgs = ConfigDict()
    c.predictor.lbfgs.maxiter = 64
    c.predictor.lbfgs.check_every = 16
    c.predictor.lbfgs.memory = 20
    c.predictor.lbfgs.max_linesearch = 30
    c.predictor.lbfgs.gtol = 1e-12
    c.predictor.lbfgs.ftol = 0.
    c.predictor.lbfgs.residual_limit = 5e-4
    c.predictor.lbfgs.range_tolerance = c.reporting.range_atol
    c.correction = ConfigDict()
    c.correction.omega = .8
    c.correction.maxiter = 2048
    c.correction.check_every = 16
    c.correction.residual_limit = 1e-5
    c.correction.range_atol = c.reporting.range_atol
    return c
