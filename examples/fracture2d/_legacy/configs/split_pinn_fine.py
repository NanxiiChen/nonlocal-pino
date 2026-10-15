"""Independent FP64 split-horizon PINN; reference fields are comparisons only."""

from ml_collections import ConfigDict
from examples.fracture2d._legacy.configs.scale_separation_fine import get_config as model_config
from examples.fracture2d._legacy.configs.train_base import get_config as neural_config


def get_config():
    c = model_config()
    base = neural_config()
    c.arch = base.arch
    c.train = base.train
    c.train.dtype = "float64"
    c.train.learning_rate = 1e-4
    c.train.mechanics_objective = "energy"
    c.physical.damage_floor = 1e-8
    c.data.reference_root = "/root/autodl-tmp/data/fracture2d/scale_separation_20261001/reference_h004_du008_budget1000"
    c.data.load_step = 0.0001
    c.data.end_state_index = 2
    c.stagger = ConfigDict()
    c.stagger.sweeps = 3
    c.stagger.mechanics_steps = 2000
    c.stagger.damage_steps = 2000
    c.stagger.omega = 0.5
    c.stagger.accept_on_first_passing_report = True
    c.reporting = ConfigDict()
    c.reporting.damage_rmse_limit = 0.01
    c.reporting.reaction_relative_limit = 0.05
    c.reporting.mechanics_loss_limit = 0.01
    c.reporting.coupled_AT2_relative_limit = 1e-3
    c.reporting.damage_change_limit = 1e-5
    c.reporting.displacement_relative_limit = 0.05
    c.reporting.healing_limit = 1e-3
    c.reporting.range_atol = 1e-10
    c.output.dir = "/root/autodl-tmp/tf-logs/fracture2d/pinn_split_fine_energy_firstpass"
    c.output.log_every = 200
    c.output.resume = False
    c.output.fork_from = ""
    c.output.fork_after_index = 0
    return c
