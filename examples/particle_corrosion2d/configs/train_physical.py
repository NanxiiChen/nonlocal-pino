"""SI KKS material,50x30um geometry and a derived unit-time PDE."""
from ml_collections import ConfigDict

from examples.particle_corrosion2d.configs.train_online import get_config as online_config
from examples.particle_corrosion2d.dimensional import derive_config


def get_config():
    c = online_config()
    c.physical = ConfigDict(dict(D=8.5e-10, A=5.35e7, L=2e-6,
        alpha_phi=1.03e-4/4, omega_phi=1.76e7, c_se=1.0,
        c_le=5100.0/1.43e5, kappa=1.0))
    c.nondim = ConfigDict(dict(L_c=50e-6, T_c=5.0))
    c.geometry_si = ConfigDict(dict(length=50e-6, height=30e-6,
        pit_x=25e-6, pit_radius=6e-6, mouth_half_width=2.75e-6))
    c.sample.Nt = 32
    c.operator.mouth_sampling = 'mesh_dirichlet'
    return derive_config(c)
