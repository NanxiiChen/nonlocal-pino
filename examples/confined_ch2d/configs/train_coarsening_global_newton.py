"""Unchanged global R/t objective, with matrix-free Newton-CG refinement."""
from ml_collections import ConfigDict
from examples.confined_ch2d.configs.train_coarsening_global_average import get_config as baseline


def get_config():
    c=baseline();old=c.train.refinement
    c.train.refinement=ConfigDict(dict(optimizer='newton_cg',iterations=200,
        time_nodes=old.time_nodes,time_points=old.time_points,
        step_tolerance=1e-8,c1=1e-4,c2=.9))
    return c
