"""Configuration for 2D FEM KKS electropolishing solver — rough interface.

Domain: [0, 1] × [0, 0.5].  Top: liquid Dirichlet (φ=0, c=0).
Bottom: solid Dirichlet (φ=1, c=1).  Left/Right: Neumann (zero flux).
Initial interface: rough tanh profile at y ≈ 0.33.
"""

from ml_collections import config_dict


def get_config():
    c = config_dict.ConfigDict()

    # --- Mesh ---
    c.mesh = config_dict.ConfigDict()
    c.mesh.nx = 200       # cells in x (0 → 1)
    c.mesh.ny = 100       # cells in y (0 → 0.5)

    # --- Physical parameters ---
    c.physical = config_dict.ConfigDict()
    c.physical.D = 8.5e-10          # diffusivity [m²/s]
    c.physical.A = 5.35e7           # coupling constant [J/m³]
    c.physical.L = 1e-6             # interface mobility
    c.physical.alpha_phi = 1.03e-4/4  # gradient energy coefficient [J/m]
    c.physical.omega_phi = 1.76e7   # double-well barrier height [J/m³]
    c.physical.c_se = 1.0
    c.physical.c_le = 5100.0 / 1.43e5

    # --- Non-dimensionalization ---
    c.nondim = config_dict.ConfigDict()
    c.nondim.L_c = 100e-6           # characteristic length [m]
    c.nondim.T_c = 5.0             # characteristic time [s]

    # --- Time stepping (non-dimensional) ---
    c.time = config_dict.ConfigDict()
    c.time.dt = 0.01
    c.time.t_final = 1.0
    c.time.theta = 0.5              # 1.0=backward Euler, 0.5=Crank-Nicolson

    # --- Solver ---
    c.solver = config_dict.ConfigDict()
    c.solver.atol = 1e-8
    c.solver.rtol = 1e-6
    c.solver.max_iters = 20

    # --- Rough surface parameters ---
    c.surface = config_dict.ConfigDict()
    c.surface.y0 = 0.33              # mean interface height (≈2/3 of 0.5)
    c.surface.n_modes = 6            # number of Fourier modes
    c.surface.amplitude = 0.03       # max amplitude of roughness
    c.surface.freq_lo = 1.0          # min spatial frequency
    c.surface.freq_hi = 5.0          # max spatial frequency
    c.surface.seed = 42              # RNG seed for reproducibility

    # --- Output ---
    c.output = config_dict.ConfigDict()
    c.output.dir = "/root/autodl-tmp/data/electropolish2d"
    c.output.filename = "results_rough.npz"
    c.output.save_every = 0.05

    return c
