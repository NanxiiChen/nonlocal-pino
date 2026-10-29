"""Global mixed fields: shared trunk, explicit mu scale and chemical-unit losses."""
from examples.confined_ch2d.configs.train_coarsening_global_strong import get_config as baseline


def get_config():
    c=baseline();c.formulation.chemical='learned'
    c.arch.model='global_mixed';c.arch.field_layout='shared'
    c.arch.chemical_scale=.5
    c.loss.chemical_weight=1.;c.loss.chemical_scale=.5
    return c
