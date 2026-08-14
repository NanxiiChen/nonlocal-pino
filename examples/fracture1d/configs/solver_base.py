"""Configuration for the 1D phase-field fracture FEniCS solver.

Domain: 1D bar x ∈ [0, 1] under uniaxial tension.
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
    c.physical.Gc = 2.7         # fracture toughness [N/mm]
    c.physical.l0 = 0.024       # internal length [mm]
    c.physical.k_res = 1.0e-8   # additive residual stiffness

    # --- Dimensional scale for the non-dimensional domain [0, 1] ---
    c.geometry = config_dict.ConfigDict()
    c.geometry.L0 = 1.0         # characteristic bar length [mm]

    # --- Loading (non-dimensional displacement control) ---
    # The base FEM reference uses the fixed increment du=1e-5. The solver
    # still supports adaptive cutbacks when these bounds are overridden.
    c.loading = config_dict.ConfigDict()
    c.loading.du = 1e-5         # fixed displacement increment
    c.loading.du_min = 1e-5     # fixed increment: no automatic cutback
    c.loading.du_max = 1e-5     # fixed increment: no automatic growth
    c.loading.du_softening = 1e-5  # fixed increment on the softening branch
    c.loading.cutback_factor = 0.5
    c.loading.growth_factor = 1.0
    c.loading.max_cutbacks = 10
    c.loading.u_final = 0.02    # final applied displacement

    # --- Modulus imperfection on x_star in [0, 1] ---
    # e(x_star) = 1 - amplitude * exp(-(x_star-center)^2/(2*width^2)).
    # center and width are non-dimensional coordinates on [0, 1].
    c.imperfection = config_dict.ConfigDict()
    c.imperfection.amplitude = 0.1    # 10% local modulus reduction
    c.imperfection.width = 0.05
    c.imperfection.center = 0.5

    # --- Solver ---
    c.solver = config_dict.ConfigDict()
    # The localized branch requires the observed 1e-6 fixed-point tolerance.
    c.solver.atol_u = 1e-6
    c.solver.atol_d = 1e-6
    c.solver.atol_H = 1e-6
    c.solver.atol_sigma = 1e-6
    c.solver.atol_stress_constancy = 1e-5
    c.solver.linear_solver = "lu"
    c.solver.max_stagger_iters = 200

    # --- Output ---
    c.output = config_dict.ConfigDict()
    c.output.log_every = 100         # progress print interval (steps)
    c.output.dir = "/root/autodl-tmp/data/fracture1d"
    c.output.filename = "results_fracture1d.npz"
    c.output.save_every = 1e-5        # save every accepted FEM load step
    c.output.plot_dir = "/root/autodl-tmp/data/fracture1d"

    return c
