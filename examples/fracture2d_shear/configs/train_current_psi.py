"""Trial: instantaneous psi+ energy plus L1 irreversibility, replay OFF."""
from ml_collections import ConfigDict
from examples.fracture2d_shear.configs.train_base import get_config as base_config


def get_config():
    c = base_config()
    c.irreversibility = ConfigDict(dict(mode='current_psi_l1', beta=1., healing_tolerance=1e-3))
    c.output.root = '/root/autodl-tmp/tf-logs/fracture2d_shear/pinn_current_psi_l1_beta1'
    c.output.event_dir = c.output.root + '/events'
    return c
