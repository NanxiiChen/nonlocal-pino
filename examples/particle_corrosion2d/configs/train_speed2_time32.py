"""Cold speed2/node-mouth control with32fresh uniform times at each update."""
from examples.particle_corrosion2d.configs.train_speed2_mouth_nodes import get_config as node_config


def get_config():
    c = node_config()
    c.sample.Nt = 32
    return c
