"""Original PDE/global network with expanding random-time coverage."""
from ml_collections import ConfigDict
from examples.particle_corrosion2d.configs.train_physical_single_joint import get_config as joint_config


def get_config():
    c=joint_config()
    c.sample.horizon_curriculum=ConfigDict(dict(kind='exponential',initial_fraction=.02,ramp_steps=4000))
    return c
