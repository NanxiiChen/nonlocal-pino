"""Same global mixed problem; independent spatial MLP for each physical field."""
from examples.confined_ch2d.configs.train_coarsening_global_mixed import get_config as baseline


def get_config():
    c=baseline();c.arch.field_layout='separate'
    return c
