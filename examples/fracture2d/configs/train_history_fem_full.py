"""Full zero-to-0.006 FE-energy history-network trial; pilot settings retained."""
from examples.fracture2d.configs.train_history_fem import get_config as pilot_config


def get_config():
    c = pilot_config()
    c.loading.u_final = .0060
    c.output.root = '/root/autodl-tmp/tf-logs/fracture2d/runs/pinn_history_fem_full'
    c.output.event_dir = c.output.root + '/events'
    c.output.reference_root = '/root/autodl-tmp/tf-logs/fracture2d/runs/fem_damage_reference_full'
    return c
