"""Training configuration for the nonlocal 1D fracture PINN."""

from ml_collections import config_dict


def get_config():
    c = config_dict.ConfigDict()

    # Damage-only network with unconstrained output, matching the direct solve.
    c.arch = config_dict.ConfigDict()
    c.arch.type = "mlp"
    c.arch.hidden_dim = 64
    c.arch.num_layers = 4
    c.arch.activation = "tanh"
    c.arch.use_spatial_emb = True
    c.arch.train_spatial_embedding = True
    c.arch.use_temporal_emb = False
    c.arch.spatial_emb_dim = 32
    c.arch.spatial_scale = 4.0
    c.arch.temporal_emb_dim = 8
    c.arch.temporal_scale = 1.0
    c.arch.film_hidden_dim = 16
    c.arch.film_num_layers = 1
    c.arch.out_dim = 1
    c.arch.spatial_dim = 1
    c.arch.tensor_split = False
    c.arch.multi_head = False
    c.arch.damage_output = "linear"
    c.arch.d_init = 1.0e-6
    c.arch.local_equilibrium = False
    c.arch.use_local_gaussian = False
    c.arch.local_gaussian_width = 1.0 / (1601 - 1)
    c.arch.local_gaussian_amplitude = 0.0
    c.arch.use_local_hat_enrichment = False
    c.arch.local_hat_count = 33

    # Dimensional material values and the dimensional length scale.
    c.physical = config_dict.ConfigDict()
    c.physical.E = 210.0e3
    c.physical.Gc = 2.7
    c.physical.l0 = 0.024
    c.physical.k_res = 1.0e-8

    c.geometry = config_dict.ConfigDict()
    c.geometry.L0 = 1.0

    # Modulus imperfection on x_star in [0, 1].
    c.imperfection = config_dict.ConfigDict()
    c.imperfection.amplitude = 0.1
    c.imperfection.width = 0.05
    c.imperfection.center = 0.5

    # SparseD is constructed once on the full fixed grid.
    c.nl = config_dict.ConfigDict()
    c.nl.delta = 0.006
    c.nl.kernel = "cubic"
    c.nl.method = "sparse"

    c.sample = config_dict.ConfigDict()
    c.sample.Nx = 1601

    c.train = config_dict.ConfigDict()
    # Match the accepted FEM reference path before testing nonlocal effects.
    c.train.du = 1.0e-5
    c.train.du_min = 1.0e-5
    c.train.du_max = 1.0e-5
    c.train.du_softening = 1.0e-5
    c.train.cutback_factor = 0.5
    c.train.growth_factor = 1.0
    c.train.max_cutbacks = 10
    c.train.u_final = 0.02
    c.train.max_stagger_iters = 200
    c.train.damage_iterations = 20000
    c.train.damage_check_every = 50
    # Each fixed-history damage solve starts with fresh optimizer moments while
    # retaining the converged network parameters as its warm start.
    c.train.reset_optimizer_per_fixed_history = True
    # Both the neural fixed-point change and the damage-equation residual are
    # required before an outer staggered state is accepted.
    c.train.atol_damage_change = 5.0e-6
    c.train.atol_damage_residual = 5.0e-6
    c.train.atol_damage_residual_inf = 2.0e-5
    # When Jacobi scaling is enabled, these are the fixed-history acceptance
    # tolerances. Raw residuals remain diagnostics and are saved separately.
    c.train.atol_damage_residual_scaled = 5.0e-6
    c.train.atol_damage_residual_scaled_inf = 2.0e-5
    c.train.atol_sigma = 1.0e-6
    c.train.atol_history = 1.0e-6
    c.train.atol_mechanics = 1.0e-6
    c.train.atol_stress_constancy = 1.0e-6
    c.train.atol_healing_diagnostic = 1.0e-6
    c.train.lr_init = 2.0e-3
    c.train.lr_decay_every = 2000
    c.train.lr_decay_rate = 0.95
    c.train.lr_min = 1.0e-5
    c.train.optimizer = "adam"
    # Training mode: warning advances finite candidates for curve diagnostics;
    # use "strict" for fixed-point validation with rollback/cutback.
    c.train.gate_mode = "warning"
    # RPROP diagnostic settings from Manav et al. (2024), retained separately
    # from the Adam/SOAP learning-rate schedule.
    c.train.rprop_initial_step_size = 1.0e-5
    c.train.rprop_eta_minus = 0.5
    c.train.rprop_eta_plus = 1.2
    c.train.rprop_min_step_size = 1.0e-10
    c.train.rprop_max_step_size = 50.0
    c.train.grad_clip = 1000.0
    c.train.dtype = "float64"
    c.train.matmul_precision = "highest"
    c.train.soap_max_precond_dim = 100000
    c.train.soap_b1 = 0.90
    c.train.soap_b2 = 0.99
    c.train.soap_precond_freq = 5
    c.train.soap_precond_1d = False
    c.train.log_every = 50
    c.train.eval_every = 10
    c.train.checkpoint_every = 50

    c.loss = config_dict.ConfigDict()
    c.loss.objective = "residual"
    c.loss.residual_norm = "mae"
    c.loss.residual_scaling = "jacobi"
    # Default-off residual-only diagnostic: emphasize the current largest rows
    # without using reference fields or fixed spatial locations.
    c.loss.hard_point_count = 16
    c.loss.hard_point_weight = 0.0
    c.loss.hard_point_mode = "dynamic"
    c.loss.hard_point_temperature = 4.0

    c.output = config_dict.ConfigDict()
    c.output.log_dir = "/root/autodl-tmp/tf-logs/fracture1d"
    c.output.run_name = ""
    # All PINN events, configs, checkpoints, and result archives are stored in
    # one run-specific directory below this TensorBoard root.
    c.output.checkpoint_dir = "/root/autodl-tmp/tf-logs/fracture1d"
    c.output.save_dir = "/root/autodl-tmp/tf-logs/fracture1d"
    c.output.filename = "results_nonlocal_pinn.npz"

    c.reference = config_dict.ConfigDict()
    c.reference.data_path = (
        "/root/autodl-tmp/data/fracture1d/classical_nx2000/"
        "results_classical_nx2000.npz"
    )
    c.reference.mesh_nx = 2000
    c.reference.history_space = "DG0_classical_max"
    c.reference.history_method = "Miehe_max_history"

    c.baseline = config_dict.ConfigDict()
    c.baseline.data_path = (
        "/root/autodl-tmp/data/fracture1d/nonlocal_baseline/"
        "results_direct_cellhistory_nx1601_metricfix.npz"
    )
    c.baseline.mesh_nx = 1601
    c.baseline.history_mode = "cell_mass_lumped"
    c.baseline.history_space = "nodal_mass_lumped_from_DG0"
    c.baseline.history_method = "Miehe_max_history_DG0_cell_strain"
    c.baseline.baseline_method = "direct_sparse_linear"
    c.baseline.max_stagger_iters = 200
    c.baseline.atol_damage = 1.0e-6

    return c
