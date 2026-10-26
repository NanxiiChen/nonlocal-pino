"""Local AD with joint random spacetime, otherwise the local tensor control."""
from examples.corrosion2d.configs.train_local_tensor import get_config as base


def get_config():
    c=base()
    c.experiment.spatial_operator="local_ad_joint"
    return c
