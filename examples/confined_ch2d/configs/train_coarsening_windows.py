"""Causal method-of-lines PINN; frozen preceding states, no reference labels."""
from ml_collections import ConfigDict
from examples.confined_ch2d.configs.train_coarsening import get_config as baseline


def get_config():
    c=baseline();c.arch.spatial_modes=8
    c.formulation=ConfigDict(dict(chemical='eliminated'));del c.loss.chemical_weight
    c.sample.include_initial_time=True
    c.train.windows=ConfigDict(dict(count=30,spacing='logarithmic',time_scale=.005,
        first_soap=650,soap_each=150,first_lbfgs=36,lbfgs_each=16,
        first_maxfun=72,maxfun_each=32,reset_optimizer=True,warm_start=True,
        metric='biharmonic_resolvent'))
    return c
