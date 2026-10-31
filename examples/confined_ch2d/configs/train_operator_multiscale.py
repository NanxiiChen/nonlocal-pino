"""Twenty-four balanced ambient modes and longerT3 coarsening; reference-source bank."""
from examples.confined_ch2d.configs.train_operator_richer import get_config as baseline


def get_config():
    c=baseline()
    c.physical.terminal_time=3.
    c.initial.amplitude=.20
    c.initial.waves=((1.,0.,.20,.2),(0.,1.,.20,1.1),
        (1.,1.,.20,.7),(1.,-1.,.20,2.2),(2.,0.,.20,1.4),(0.,2.,.20,.4),
        (2.,-1.,.20,2.8),(2.,1.,.20,1.9),(1.,2.,.18,3.2),(1.,-2.,.18,.9),
        (2.,2.,.18,2.5),(2.,-2.,.18,4.1),(3.,0.,.18,2.),(0.,3.,.18,4.8),
        (3.,1.,.18,.5),(1.,3.,.18,3.6),(3.,-1.,.18,5.3),(1.,-3.,.18,1.7),
        (3.,2.,.18,4.3),(2.,3.,.18,5.8),(3.,-2.,.18,.35),(2.,-3.,.18,2.65),
        (3.,3.,.18,4.6),(3.,-3.,.18,5.9))
    c.operator.dataset_path='/root/autodl-tmp/data/confined_ch2d/operator_geometry_multiscale_v1'
    c.operator.reference_workers=6
    c.operator.local_reference_workers=2
    c.train.iterations=20000
    return c
