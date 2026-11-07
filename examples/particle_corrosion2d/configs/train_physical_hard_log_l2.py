"""Global log-time initial lifting with conventional squared PDE residuals."""
from examples.particle_corrosion2d.configs.train_physical_hard_log import get_config as log_config


def get_config():
    c = log_config()
    c.loss.pde_reduction = 'mean_square'
    return c
