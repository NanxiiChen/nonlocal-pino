"""Cold L-panel AT2/history trial; same material and neural scheme as shear."""
from ml_collections import ConfigDict
from examples.fracture2d_shear.configs.train_base import get_config as shear_config


def get_config():
    c = shear_config()
    c.boundary = ConfigDict(dict(mode='l_shaped_panel', corner_x=.5, corner_y=.5,
                                load_start=.94, load_end=1.))
    c.geometry.width = 1.
    c.geometry.height = 1.
    c.geometry.mesh_size = .004
    c.data.cloud = '/root/autodl-tmp/data/fracture2d_lshape/cloud.npz'
    c.mesh.prepared_dir = '/root/autodl-tmp/data/fracture2d_lshape/geometry_assets'
    c.loading.u_final = .0300
    c.history.replay_capacity = 1024
    c.history.replay_enabled = False
    c.output.root = '/root/autodl-tmp/tf-logs/fracture2d_lshape/pinn_history_conditioned'
    c.output.event_dir = c.output.root + '/events'
    c.output.reference_root = '/root/autodl-tmp/data/fracture2d_lshape/fem_reference'
    return c
