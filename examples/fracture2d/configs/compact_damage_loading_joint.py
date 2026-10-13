"""Train conditioned trunk and compact readouts together on own-zero H."""
from examples.fracture2d.configs.compact_damage_loading import get_config as base


def get_config():
    c = base()
    c.loading.output_dir = '/root/autodl-tmp/tf-logs/fracture2d/compact_damage_loading_joint_20261011'
    c.loading.train_trunk = True
    c.loading.lbfgs.maxiter = 2048
    c.loading.lbfgs.check_every = 32
    return c
