"""Configuration for the FEM phase-field corrosion solver."""

from ml_collections import config_dict


def get_config():
    config = config_dict.ConfigDict()

    # --- Mesh ---
    config.mesh = config_dict.ConfigDict()
    config.mesh.n_cells = 250

    # --- Physical parameters ---
    config.physical = config_dict.ConfigDict()
    config.physical.D = 8.5e-10          # diffusivity [m²/s]
    config.physical.A = 5.35e7           # coupling constant [J/m³]
    config.physical.L = 1e-11              # interface mobility (diffusion-controlled)
    config.physical.alpha_phi = 1.03e-4  # gradient energy coefficient [J/m]
    config.physical.omega_phi = 1.76e7   # double-well barrier height [J/m³]
    config.physical.c_se = 1.0           # equilibrium concentration in solid
    config.physical.c_le = 5100.0 / 1.43e5  # equilibrium concentration in electrolyte

    # --- Non-dimensionalization ---
    config.nondim = config_dict.ConfigDict()
    config.nondim.L_c = 100e-6           # characteristic length = domain size [m]
    config.nondim.T_c = 1e5             # characteristic time = t_final [s]

    # --- Time stepping (non-dimensional) ---
    config.time = config_dict.ConfigDict()
    config.time.dt = 0.005
    config.time.t_final = 1.0
    config.time.theta = 0.5  # 1.0=backward Euler, 0.5=Crank-Nicolson

    # --- Non-local operator ---
    config.nl = config_dict.ConfigDict()
    config.nl.delta = 0.02
    config.nl.kernel = "cubic"  # "cubic" or "tophat"

    # --- Solver ---
    config.solver = config_dict.ConfigDict()
    config.solver.atol = 1e-8
    config.solver.rtol = 1e-6
    config.solver.max_iters = 10

    # --- Initial condition ---
    config.init = config_dict.ConfigDict()
    config.init.center = 0.5               # interface center in x* coordinates

    # --- Output ---
    config.output = config_dict.ConfigDict()
    config.output.dir = "/root/autodl-tmp/data/corrosion1d"
    config.output.filename = "results_activation_cn.npz"

    return config
