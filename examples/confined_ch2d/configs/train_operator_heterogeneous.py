"""Generic boundary/function point streams; stronger same-bank CH comparison."""
from ml_collections import ConfigDict
from examples.confined_ch2d.configs.train_operator_longer import get_config as baseline


def get_config():
    c=baseline()
    c.arch=ConfigDict(dict(model='heterogeneous',input_dims=(5,3),extent=(1.,1.),
        width=64,heads=4,condition_depth=4,decoder_depth=3,spatial_features=24,
        spatial_scale=1.,time_features=4,output_scale=.1,norm_eps=1e-5))
    return c
