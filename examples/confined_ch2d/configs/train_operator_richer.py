"""Eight fixed IC waves and broad random holes on a modestly finer mesh."""
from examples.confined_ch2d.configs.train_operator_heterogeneous import get_config as baseline


def get_config():
    c=baseline()
    c.geometry.mesh_size=.035
    c.initial.amplitude=.22
    c.initial.waves=((1.,0.,.25,.2),(0.,1.,.25,1.1),
        (1.,1.,.65,.7),(1.,-1.,.6,2.2),(2.,0.,.5,1.4),
        (0.,2.,.4,.4),(2.,-1.,.4,2.8),(2.,1.,.35,1.9))
    c.operator.center_min=((.22,.27),(.22,.27))
    c.operator.center_max=((.78,.73),(.78,.73))
    c.operator.radius_min=(.045,.045)
    c.operator.radius_max=(.12,.12)
    c.operator.dataset_path='/root/autodl-tmp/data/confined_ch2d/operator_geometry_richer_v1'
    return c
