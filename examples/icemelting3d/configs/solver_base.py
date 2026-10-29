"""Explicit local FEM controls; physical/IC data come from saved PINO config."""
from ml_collections import ConfigDict


def get_config():
    c = ConfigDict()
    c.reference = ConfigDict(dict(
        training_config='/root/autodl-tmp/tf-logs/icemelting3d/20261029-melting-operator-online-b16-01/config.json',
        manifest='/root/autodl-tmp/data/icemelting3d/operator_ellipsoids_v1/manifest.json',
        root='/root/autodl-tmp/data/icemelting3d/operator_local_fem_v1',
        indices=(5,), snapshots=5, mesh_levels=(32, 48, 64, 96),
        time_steps=(.005, .0025, .00125, .000625),
        phase_tolerance=5e-4, solid_tolerance=.01))
    c.mesh = ConfigDict(dict(degree=2, quadrature_degree=8))
    c.time = ConfigDict(dict(scheme='imex_bdf2'))
    c.solver = ConfigDict(dict(linear_solver='cg', preconditioner='jacobi',
        atol=1e-11, rtol=1e-10, maximum_iterations=12,
        krylov_atol=1e-13, krylov_rtol=1e-11, krylov_maximum_iterations=200))
    return c
