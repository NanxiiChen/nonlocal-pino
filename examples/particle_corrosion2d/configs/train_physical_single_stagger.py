"""Full-duration AC/CH loss alternation against the joint control."""
from examples.particle_corrosion2d.configs.train_physical_single_joint import get_config as joint_config


def get_config():
    c=joint_config()
    c.loss.stagger_steps=c.train.iterations-c.loss.data_warmup_steps
    c.loss.stagger_switch_every=25
    return c
