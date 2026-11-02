"""Phase/flow loss blocks, lagged coupling and inactive-head update masking."""
from ml_collections import ConfigDict
from examples.channel_nsch2d.configs.train_operator import get_config as baseline


def get_config():
    c=baseline()
    c.stagger=ConfigDict(dict(kind='phase_flow',switch_every=25,alternating_steps=4000,
        coupling='lagged',inactive_head='mask_updates',optimizer_state='shared'))
    return c
