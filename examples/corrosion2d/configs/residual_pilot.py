"""Explicit matched residual experiments built on preserved historical presets."""

from ml_collections import config_dict


def get_strategy_config(case, formulation):
    if case == "corrosion2d":
        from examples.corrosion2d.configs.train_1pit import get_config as base
    elif case == "electropolish2d":
        from examples.electropolish2d.configs.train_legacy import get_config as base
    else:
        raise ValueError(f"Unknown KKS experiment case: {case}")
    if formulation not in ("phase", "latent", "bounded", "bounded_frozen"):
        raise ValueError(f"Unknown phase residual: {formulation}")
    c = base()
    c.sample.Nx, c.sample.Ny = 101, 51
    c.sample.Nt = 16
    c.sample.K_core, c.sample.K_pool = 256, 100
    c.sample.N_ic_global, c.sample.N_ic_local = 1000, 1000
    if case == "corrosion2d":
        c.sample.N_bc_dir, c.sample.N_bc_solid = 128, 128
    else:
        c.sample.N_bc_top, c.sample.N_bc_bottom = 128, 128
    c.nl.method = "sparse"
    c.train.iterations = 5000
    c.train.lr_init = 1e-3
    c.train.optimizer = "soap"
    c.train.compute_dtype = "float32"  # promote common FP32 inputs for FP64 controls
    c.train.gpu_policy = "deterministic"  # historical research baseline; canonical presets override
    c.train.gradnorm_update_every = 25
    c.loss.iface_gamma_init = 0.0
    c.loss.iface_gamma_decay_steps = 0
    c.loss.iface_gamma_decay_to = 0.0
    c.loss.use_causal = False
    c.loss.stagger_steps = 0
    c.loss.t_ramp_steps = 0
    c.log.run_name = ""
    c.log.log_every = 100
    c.log.eval_every = 200
    c.experiment = config_dict.ConfigDict()
    c.experiment.case = case
    c.experiment.spatial_operator = "nonlocal_graph"
    c.experiment.local_reference_path = "/root/autodl-tmp/data/corrosion2d/results_local_ad_20261026.npz"
    c.experiment.formulation = formulation
    c.experiment.reference_path = (
        f"/root/autodl-tmp/data/{case}/results_matched_graph_20261024.npz"
    )
    c.experiment.reference_rtol = 1e-7
    c.experiment.reference_atol = 1e-10
    c.experiment.reference_max_step = 0.01
    c.experiment.refined_rtol = 1e-8
    c.experiment.refined_atol = 1e-11
    c.experiment.refined_max_step = 0.005
    c.experiment.reference_snapshots = 21
    c.experiment.reference_error_tolerance = 1e-4
    c.experiment.latent_normalization = 0.5
    c.experiment.bounded_eta = 0.1  # research weight cap11; unused by phase/latent
    c.experiment.data_loss_mode = "physical"  # physical, s_bc, s_bc_ic
    c.experiment.initial_sampling = "random_interior"  # or matched_graph
    c.experiment.boundary_sampling = "random"  # or matched_graph
    c.experiment.transport_evaluation = "physical_difference"  # or kks_raw_edges
    c.experiment.ch_norm = "l1"  # or rms; only the CH reduction changes
    c.experiment.ch_rms_floor = 1e-12  # explicit smooth zero for RMS
    c.experiment.lr_tail_start = 0  # 0 keeps the legacy schedule; research-only
    c.experiment.lr_tail_end = 1e-5  # rate at the end of the fixed update budget
    c.experiment.ch_rms_after = 0  # 0 keeps one norm; otherwise L1 then RMS
    c.experiment.checkpoint_every = 1000
    return c


def get_config(case, formulation, seed):
    """Legacy research-run preset; preserve its explicit seed and run identity."""
    c = get_strategy_config(case, formulation)
    c.train.seed = seed
    c.log.run_name = f"20261024-residual-{formulation}-seed{seed}"
    if formulation == "bounded":
        c.log.run_name = f"20261025-residual-bounded-eta{c.experiment.bounded_eta}-seed{seed}"
    return c
