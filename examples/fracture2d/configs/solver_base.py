"""Configuration for the local FEM fracture2d reference solver.

The dimensional units follow the accepted fracture1d reference: the unit
square has the dimensional length scale ``L0=1`` and material values are in
the corresponding consistent units.
"""

from ml_collections import config_dict


def get_config():
    c = config_dict.ConfigDict()

    c.geometry = config_dict.ConfigDict()
    c.geometry.Lx = 1.0
    c.geometry.Ly = 1.0
    c.geometry.L0 = 1.0
    c.geometry.notch_tip_x = 0.5
    c.geometry.notch_center_y = 0.5
    c.geometry.notch_width = 0.048
    c.geometry.notch_tip_radius = 0.024

    c.physical = config_dict.ConfigDict()
    c.physical.E = 210.0e3
    c.physical.nu = 0.3
    c.physical.Gc = 2.7
    c.physical.l0 = 0.024
    c.physical.k_res = 1.0e-8

    c.boundary = config_dict.ConfigDict()
    c.boundary.mode = "bottom_vertical_point_horizontal"
    c.boundary.anchor_x = 0.0
    c.boundary.anchor_y = 0.0

    c.mesh = config_dict.ConfigDict()
    c.mesh.geo_file = "examples/fracture2d/mesh.geo"
    c.mesh.msh_file = "/root/autodl-tmp/data/fracture2d/local_fem/rounded_slot.msh"
    c.mesh.xdmf_file = "/root/autodl-tmp/data/fracture2d/local_fem/rounded_slot.xdmf"
    c.mesh.lc_far = 0.02
    c.mesh.lc_near = 0.004
    c.mesh.rebuild = True

    c.loading = config_dict.ConfigDict()
    c.loading.du = 1.0e-4
    c.loading.du_min = 1.0e-5
    c.loading.du_max = 1.0e-4
    c.loading.du_softening = 2.5e-5
    c.loading.cutback_factor = 0.5
    c.loading.growth_factor = 1.0
    c.loading.max_cutbacks = 8
    c.loading.u_final = 0.0075

    c.solver = config_dict.ConfigDict()
    c.solver.max_stagger_iters = 100
    # Warning mode commits finite candidates after the iteration budget, as in
    # the accepted 1D curve-diagnostic baseline. Set to "strict" for rollback.
    c.solver.gate_mode = "warning"
    c.solver.atol_u = 1.0e-8
    c.solver.atol_d = 1.0e-5
    # The absolute history gate is relaxed relative to the 1D strict pilot.
    # Absolute and scale-normalized history defects are both archived.
    c.solver.atol_history = 1.0e-5
    c.solver.rtol_history = 1.0e-5
    c.solver.atol_reaction = 1.0e-4
    c.solver.rtol_reaction = 1.0e-6
    c.solver.max_newton_iters = 30
    c.solver.newton_atol = 1.0e-8
    c.solver.newton_rtol = 1.0e-8
    c.solver.newton_relaxation = 1.0
    c.solver.linear_solver = "lu"

    c.output = config_dict.ConfigDict()
    c.output.log_every = 10
    c.output.save_every = 1.0e-4
    c.output.dir = "/root/autodl-tmp/data/fracture2d/local_fem_point_anchor"
    c.output.filename = "results_point_anchor.npz"
    c.output.plot_dir = "/root/autodl-tmp/data/fracture2d/local_fem_point_anchor"
    c.output.field_grid_loads = [0.0055, 0.0058, 0.0059, 0.0060, 0.0061, 0.0062]

    return c
