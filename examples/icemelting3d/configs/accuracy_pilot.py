"""Explicit, isolated accuracy pilot; does not change production defaults."""

from ml_collections import ConfigDict

from examples.icemelting3d.configs.train_base import get_config as base_config


def get_config():
    c = base_config()
    c.nondim.T_c = 5.0
    c.train.iterations = 2000
    c.train.lr_init = 1e-3
    c.sample.K_core = 256
    c.sample.Nt = 16
    c.sample.N_ic = 4096
    c.experiment = ConfigDict()
    c.experiment.seed = 42
    c.experiment.output_dir = "reports/icemelting-accuracy-20261024"
    c.experiment.reference_dt = 0.001
    c.experiment.reference_snapshots = 21
    c.experiment.eval_chunk = 32768
    c.experiment.log_every = 200
    c.experiment.curriculum_stages = 5
    c.experiment.methods = [
        "phase_soft", "latent_phase", "latent_balanced", "latent_curriculum"
    ]
    return c
