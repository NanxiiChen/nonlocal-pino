"""Experimental conservative CH potential; full coordinates, same online prior."""
from examples.confined_ch2d.configs.train_operator import get_config as baseline


def get_config():
    c=baseline()
    c.operator.formulation='conservative_potential'
    c.operator.metric='identity'
    c.train.iterations=5000
    c.log.checkpoint_every=5000
    c.log.run_name=''
    return c
