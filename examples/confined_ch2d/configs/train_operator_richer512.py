"""512 geometry operators, unchanged physical case and eight validation domains."""
from ml_collections import ConfigDict
from examples.confined_ch2d.configs.train_operator_richer import get_config as baseline


def get_config():
    c=baseline()
    c.operator.train_conditions=512
    c.operator.local_validation_indices=(512,513)
    c.operator.dataset_path='/root/autodl-tmp/data/confined_ch2d/operator_geometry_richer512_v1'
    c.operator.precompute=ConfigDict(dict(workers=6,
        source_path='/root/autodl-tmp/data/confined_ch2d/operator_geometry_richer_v1',
        reserved_test_draws=tuple(range(16,24)),monitor_train_indices=tuple(range(16))))
    return c
