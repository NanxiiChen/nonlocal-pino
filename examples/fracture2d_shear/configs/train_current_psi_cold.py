"""Cold shear trial with screened beta10 L1 irreversibility; replay OFF."""
from examples.fracture2d_shear.configs.train_current_psi import get_config as penalty_config


def get_config():
    c = penalty_config()
    c.irreversibility.beta = 10.
    c.output.root = '/root/autodl-tmp/tf-logs/fracture2d_shear/pinn_current_psi_l1_beta10_cold'
    c.output.event_dir = c.output.root + '/events'
    return c
