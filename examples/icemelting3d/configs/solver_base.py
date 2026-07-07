"""Ice melting 3D FEM solver config (non-dimensional)."""

from ml_collections import config_dict


def get_config():
    c = config_dict.ConfigDict()

    c.mesh = config_dict.ConfigDict()
    c.mesh.N = 64            # elements per direction (N³ cells)

    c.time = config_dict.ConfigDict()
    c.time.dt = 0.001        # non-dimensional time step dt* (= 0.005 / T_c)
    c.time.T = 1.0           # total non-dimensional time T* = 1
    c.time.theta = 0.5       # Crank-Nicolson (0.5); Backward Euler (1.0)

    c.physical = config_dict.ConfigDict()
    c.physical.M = 0.1       # mobility
    c.physical.lam = 5.0     # interface speed (lambda)
    c.physical.R0 = 35.0     # initial sphere radius (dimensional)

    c.nondim = config_dict.ConfigDict()
    c.nondim.L_c = 100.0     # characteristic length (= domain width)
    c.nondim.T_c = 5.0       # characteristic time (total dimensional time = 5 s)

    c.solver = config_dict.ConfigDict()
    c.solver.atol = 1e-8
    c.solver.rtol = 1e-6
    c.solver.max_iters = 20

    c.output = config_dict.ConfigDict()
    c.output.dir = "/root/autodl-tmp/data/icemelting3d/"
    c.output.filename = "icemelting3d.npz"
    c.output.save_every = 0.05

    return c
