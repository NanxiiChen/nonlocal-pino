"""Original SOTA with rolling loss rows; full queries, finite cold5k control."""
from ml_collections import ConfigDict
from examples.confined_ch2d.configs.train_operator import get_config as baseline


def get_config():
    c = baseline()
    c.sample.residual_rows = ConfigDict(dict(count=512, pools=8, refresh_every=50, workers=2))
    c.train.iterations = 5000
    c.log.run_name = ''
    return c
