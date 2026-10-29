"""Sharper, saturated multimode confined CH; explicit reference-first variant."""
from ml_collections import ConfigDict
from examples.confined_ch2d.configs.train_base import get_config as base_config


def get_config():
    c=base_config()
    c.initial=ConfigDict(dict(kind='multimode',amplitude=.98,wall_width=.03,scale=.55,
        waves=((2,1,1.,0.),(1,3,.75,.8),(3,-1,.6,1.6),(4,2,.4,2.4),
               (2,-4,.35,.4),(5,1,.25,2.1))))
    c.physical.epsilon=.02
    c.physical.terminal_time=3.
    c.geometry.mesh_size=.02
    c.nl.delta=.03
    c.reference.mesh_sizes=(.02,.014)
    c.reference.dt=.01
    c.reference.refined_dt=.005
    c.reference.root='/root/autodl-tmp/data/confined_ch2d/coarsening'
    c.reference.path=c.reference.root+'/reference.npz'
    return c
