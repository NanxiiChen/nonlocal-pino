"""Configuration for 2D FEM KKS phase-field corrosion solver — single pit."""

from ml_collections import config_dict


def get_config():
    c = config_dict.ConfigDict()

    # --- Mesh ---
    c.mesh = config_dict.ConfigDict()
    c.mesh.nx = 160       # cells in x (0 → 1)
    c.mesh.ny = 80        # cells in y (0 → 0.5)

    # --- Physical parameters ---
    c.physical = config_dict.ConfigDict()
    c.physical.D = 8.5e-10          # diffusivity [m²/s]
    c.physical.A = 5.35e7           # coupling constant [J/m³]
    c.physical.L = 2.0              # interface mobility (diffusion-controlled)
    c.physical.alpha_phi = 1.03e-4  # gradient energy coefficient [J/m]
    c.physical.omega_phi = 1.76e7   # double-well barrier height [J/m³]
    c.physical.c_se = 1.0
    c.physical.c_le = 5100.0 / 1.43e5

    # --- Non-dimensionalization ---
    c.nondim = config_dict.ConfigDict()
    c.nondim.L_c = 100e-6           # L in meters = domain length
    c.nondim.T_c = 50.0

    # --- Time stepping (non-dimensional) ---
    c.time = config_dict.ConfigDict()
    c.time.dt = 0.005
    c.time.t_final = 1.0
    c.time.theta = 1.0   # 1.0=backward Euler, 0.5=Crank-Nicolson

    # --- Solver ---
    c.solver = config_dict.ConfigDict()
    c.solver.atol = 1e-8
    c.solver.rtol = 1e-6
    c.solver.max_iters = 20

    # --- Initial pit(s): list of (cx, cy, radius) ---
    c.pits = config_dict.ConfigDict()
    c.pits.entries = ((0.5, 0.0, 0.1),)

    # --- Output ---
    c.output = config_dict.ConfigDict()
    c.output.dir = "/root/autodl-tmp/data/corrosion2d"
    c.output.filename = "results_1pit.npz"
    c.output.save_every = 0.05     # save every 0.05 non-dim time

    return c
