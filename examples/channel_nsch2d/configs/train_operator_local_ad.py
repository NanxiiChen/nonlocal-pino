"""Cold20k full spatial AD control; time blocking changes execution order only."""
from ml_collections import ConfigDict
from examples.channel_nsch2d.configs.train_default import get_config as baseline


def get_config():
    c = baseline()
    c.nl = ConfigDict(dict(method='local_ad', loss_quadrature='sbp4_mapped'))
    del c.derivatives
    del c.phase_transport
    c.train.microbatch.size = 2
    return c
