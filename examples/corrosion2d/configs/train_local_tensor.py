"""Full local-AD control, same default data/optimizer with explicit no-flux BC."""
from examples.corrosion2d.configs.train_base import get_config as base


def get_config():
    c=base()
    c.train.gpu_policy="deterministic"
    c.train.compute_dtype="float32"
    c.experiment.spatial_operator="local_ad_tensor"
    c.loss.gradnorm_bias=(1.,1.,1.,1.,1.,1.)
    return c
