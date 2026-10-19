"""Material-edge shear; cold history-conditioned FEM energy, no replay."""
from ml_collections import ConfigDict
from examples.fracture2d.configs.train_history_fem_cold_scaled import get_config as tension_config


def get_config():
    c = tension_config()
    c.boundary = ConfigDict(dict(mode='horizontal_clamped_edges'))
    c.geometry.grip_extent = 0.
    c.mesh.prepared_dir = '/root/autodl-tmp/data/fracture2d_shear/geometry_assets'
    c.loading.u_final = .0200
    c.history.replay_capacity = 512
    c.history.replay_enabled = False
    c.output.root = '/root/autodl-tmp/tf-logs/fracture2d_shear/pinn_history_conditioned'
    c.output.event_dir = c.output.root + '/events'
    c.output.reference_root = '/root/autodl-tmp/data/fracture2d_shear/fem_reference'
    c.output.resume = False
    return c
