"""2D Electropolishing PINN training config — rough interface."""

from ml_collections import config_dict


def get_config():
    c = config_dict.ConfigDict()

    # Architecture
    c.arch = config_dict.ConfigDict()
    c.arch.type = "mlp"
    c.arch.hidden_dim = 64
    c.arch.film_hidden_dim = 32
    c.arch.film_num_layers = 2
    c.arch.num_layers = 6
    c.arch.spatial_emb_dim = 64
    c.arch.spatial_scale = (1.0,3.0,6.0)
    c.arch.temporal_emb_dim = 64
    c.arch.temporal_scale = 2.0
    c.arch.activation = "gelu"
    c.arch.use_spatial_emb = True
    c.arch.use_temporal_emb = True
    c.arch.use_hard_constraint = True

    # Non-local operator
    c.nl = config_dict.ConfigDict()
    c.nl.delta = 0.02
    c.nl.kernel = "cubic"
    c.nl.method = "sparse"  # "sparse" | "fd"

    # Sampling
    c.sample = config_dict.ConfigDict()
    c.sample.Nx = 301
    c.sample.Ny = 151
    c.sample.Nt = 50
    c.sample.t_min = 0.001
    c.sample.N_bc_top = 500
    c.sample.N_bc_bottom = 500
    c.sample.N_ic_global = 2000
    c.sample.N_ic_local = 2000
    c.sample.K_core = 0
    c.sample.K_pool = 400

    # Rough surface (must match solver config for reproducibility)
    c.surface = config_dict.ConfigDict()
    c.surface.y0 = 0.33
    c.surface.n_modes = 6
    c.surface.amplitude = 0.03
    c.surface.freq_lo = 1.0
    c.surface.freq_hi = 5.0
    c.surface.seed = 42

    # Training
    c.train = config_dict.ConfigDict()
    c.train.iterations = 50000
    c.train.lr_init = 5e-3
    c.train.lr_decay_every = 2000
    c.train.lr_decay_rate = 0.95
    c.train.lr_min = 1e-5
    c.train.optimizer = "soap"
    c.train.grad_clip = 50.0
    c.train.alpha_gradnorm = 1.0
    c.train.gradnorm_update_every = 25
    c.train.soap_b1 = 0.90
    c.train.soap_b2 = 0.99
    c.train.soap_precond_freq = 5
    c.train.soap_max_precond_dim = 100000
    c.train.soap_precond_1d = False

    # Loss
    c.loss = config_dict.ConfigDict()
    c.loss.gradnorm_weight_cap = 100.0
    c.loss.gradnorm_bias = (1.0, 1.0, 1.0, 1.0, 1.0)  # ac, ch, bc_top, bc_bottom, ic
    c.loss.bias_schedule = ()
    c.loss.ac_scale = 1.0
    c.loss.ch_scale = 1.0
    c.loss.iface_gamma_init = 0.0
    c.loss.iface_gamma_decay_steps = 0
    c.loss.iface_gamma_decay_to = 0.0
    c.loss.use_causal = False
    c.loss.stagger_steps = 1000
    c.loss.stagger_switch_every = 25
    c.loss.t_ramp_steps = 0
    c.loss.t_ramp_early = 0.0
    c.loss.t_ramp_late = 1.0

    # Causal
    c.causal = config_dict.ConfigDict()
    c.causal.n_chunks = 0
    c.causal.eps_init = 1.0
    c.causal.max_eps = 100.0
    c.causal.step_size = 1.1
    c.causal.max_last_weight = 0.9
    c.causal.min_mean_weight = 0.01

    # Logging
    c.log = config_dict.ConfigDict()
    c.log.log_dir = "/root/autodl-tmp/tf-logs/electropolish2d/"
    c.log.run_name = ""
    c.log.log_every = 25
    c.log.eval_every = 200
    c.log.save_ckpt = True
    c.log.run_diag = False

    # Physical parameters
    c.physical = config_dict.ConfigDict()
    c.physical.D = 8.5e-10
    c.physical.A = 5.35e7
    c.physical.L = 1e-6
    c.physical.alpha_phi = 1.03e-4 / 4
    c.physical.omega_phi = 1.76e7
    c.physical.c_se = 1.0
    c.physical.c_le = 5100.0 / 1.43e5

    # Non-dimensionalization
    c.nondim = config_dict.ConfigDict()
    c.nondim.L_c = 100e-6
    c.nondim.T_c = 5.0

    # FEM reference data
    c.fem = config_dict.ConfigDict()
    c.fem.data_path = "/root/autodl-tmp/data/electropolish2d/results_rough.npz"

    return c
