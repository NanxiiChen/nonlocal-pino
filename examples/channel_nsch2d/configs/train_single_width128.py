"""Cold plain four-output capacity control, finest existing space/time recipe."""
from examples.channel_nsch2d.configs.train_single_conservative import get_config as baseline


def get_config():
    c=baseline()
    c.sample.Nx=192;c.sample.Ny=80;c.time_discretization.dt=.025
    c.arch.width=128;c.arch.spatial_features=64
    c.train.iterations=20000
    c.reference.root='/root/autodl-tmp/data/channel_nsch2d/dense-reference-20261101'
    c.log.eval_every=500;c.log.checkpoint_every=5000
    return c
