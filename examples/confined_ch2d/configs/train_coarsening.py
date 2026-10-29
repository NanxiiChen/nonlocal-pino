"""Sharper, saturated multimode confined CH; explicit reference-first variant."""
from ml_collections import ConfigDict
from examples.confined_ch2d.configs.train_base import get_config as base_config


def get_config():
    c=base_config()
    c.initial=ConfigDict(dict(kind='multimode',amplitude=.98,wall_width=.03,scale=.55,
        waves=((3,1,.8,0.),(1,3,.75,.8),(3,-2,.7,1.6),(2,-4,.65,2.4),
               (4,3,.6,.4),(3,-4,.55,2.1),(5,1,.5,1.1),(1,-5,.45,2.9),
               (2,2,.4,1.9),(-2,4,.35,.6))))
    c.physical.epsilon=.02
    c.physical.terminal_time=1.5
    c.geometry.mesh_size=.02
    c.nl.delta=.03
    c.reference.mesh_sizes=(.014,.01)
    c.reference.dt=.005
    c.reference.refined_dt=.0025
    c.reference.root='/root/autodl-tmp/data/confined_ch2d/coarsening_multimode10'
    c.reference.path=c.reference.root+'/reference.npz'
    return c
