"""Cold5k temporal-coverage control;40random times and one SOAP update."""
from ml_collections import ConfigDict
from examples.channel_nsch2d.configs.train_operator_width128_gmls_first_online import get_config as baseline


def get_config():
    c=baseline()
    c.sample.Nt=40
    c.train.microbatch=ConfigDict(dict(kind='time',size=8))
    c.train.iterations=5000
    return c
