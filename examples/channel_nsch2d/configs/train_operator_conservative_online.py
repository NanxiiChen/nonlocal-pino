"""Untrained cold online-geometry extension; enable after single-geometry gates."""
from ml_collections import ConfigDict
from examples.channel_nsch2d.configs.train_single_conservative import get_config as baseline


def get_config():
    c=baseline();c.sample.Nx=192;c.sample.Ny=80
    c.operator.train_conditions=16;c.operator.condition_batch=2
    c.operator.validation_manifest='/root/autodl-tmp/data/channel_nsch2d/unseen-geometry16-20261101/manifest.json'
    c.operator.online=ConfigDict(dict(workers=2,refresh_size=2,refresh_every=50,prefetch_batches=2))
    c.train.iterations=20000;c.log.checkpoint_every=5000
    return c
