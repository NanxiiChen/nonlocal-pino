"""Configuration for 1D phase-field fracture FEniCS solver — non-dimensional.

Domain: 1D bar x ∈ [-0.5, 0.5] under uniaxial tension.
Left end fixed, right end displaced (displacement-controlled loading).

Non-dimensional parameters follow the fracture1d README.
"""

from ml_collections import config_dict


def get_config():
    c = config_dict.ConfigDict()

    # --- Mesh ---
    c.mesh = config_dict.ConfigDict()
    c.mesh.nx = 2000            # number of 1D elements

    # --- Physical parameters (dimensional, Miehe benchmark) ---
    c.physical = config_dict.ConfigDict()
    c.physical.E = 210.0e3      # Young's modulus [N/mm²]
    c.physical.nu = 0.3         # Poisson's ratio
    c.physical.Gc = 2.7         # fracture toughness [N/mm]
    c.physical.l0 = 0.024       # internal length [mm]

    # --- Non-dimensionalisation ---
    c.nondim = config_dict.ConfigDict()
    c.nondim.L0 = 1.0           # characteristic length [mm] (domain = 1 mm)

    # --- Loading (non-dimensional displacement control) ---
    c.loading = config_dict.ConfigDict()
    c.loading.du = 1e-5         # displacement increment per step
    c.loading.u_final = 0.05    # final applied displacement

    # --- Imperfection (to trigger localisation at x=0) ---
    # G_c_eff(x) = G_c * (1 - amplitude * exp(-x² / (2·width²)))
    c.imperfection = config_dict.ConfigDict()
    c.imperfection.amplitude = 0.1    # 10% G_c reduction at centre
    c.imperfection.width = 0.005      # half-width of imperfection

    # --- Solver ---
    c.solver = config_dict.ConfigDict()
    c.solver.atol = 1e-10
    c.solver.rtol = 1e-8
    c.solver.max_iters = 50

    # --- Output ---
    c.output = config_dict.ConfigDict()
    c.output.dir = "/root/autodl-tmp/data/fracture1d"
    c.output.filename = "results_fracture1d.npz"
    c.output.save_every = 1e-3        # save snapshots every this much u_bar
    c.output.plot_dir = "/root/autodl-tmp/data/fracture1d"

    return c
