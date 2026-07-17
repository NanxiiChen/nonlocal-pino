"""Training config for PINN on 1D KKS corrosion."""

from ml_collections import config_dict


def get_config():
    c = config_dict.ConfigDict()

    # Architecture
    c.arch = config_dict.ConfigDict()
    c.arch.type = "mlp"  # "mlp" | "modified_mlp" | "film"
    c.arch.rank_p = 50
    c.arch.hidden_dim = 64
    c.arch.film_hidden_dim = 32
    c.arch.film_num_layers = 2     # depth of FiLM conditioning MLP
    c.arch.num_layers = 4
    c.arch.spatial_emb_dim = 64
    c.arch.spatial_scale = 1.0
    c.arch.temporal_emb_dim = 64
    c.arch.temporal_scale = 1.0
    c.arch.activation = "gelu"
    c.arch.use_bias = True
    c.arch.use_spatial_emb = True
    c.arch.use_temporal_emb = True
    c.arch.use_hard_constraint = True

    # Non-local operator
    c.nl = config_dict.ConfigDict()
    c.nl.delta = 0.02
    c.nl.kernel = "cubic"  # "cubic" (smooth) or "tophat" (discontinuous)
    c.nl.method = "sparse"  # "sparse" or "dense"

    # Sampling
    c.sample = config_dict.ConfigDict()
    c.sample.Nx = 201
    c.sample.Nt = 100
    c.sample.t_min = 0.001  # exclude t=0 from PDE sampling (IC handles t=0)

    # Training
    c.train = config_dict.ConfigDict()
    c.train.matmul_precision = "highest"
    c.train.iterations = 5000
    c.train.lr_init = 5e-4
    c.train.lr_decay_every = 100
    c.train.lr_decay_rate = 0.95
    c.train.lr_min = 1e-5
    c.train.optimizer = "adam"  # "adam" or "soap"
    c.train.grad_clip = 100.0
    c.train.soap_b1 = 0.95
    c.train.soap_b2 = 0.95
    c.train.soap_precond_freq = 5
    c.train.soap_max_precond_dim = 10000
    c.train.soap_precond_1d = False
    c.train.alpha_gradnorm = 1.0

    # Loss
    c.loss = config_dict.ConfigDict()
    c.loss.ac_scale = 1e-7
    c.loss.ch_scale = 1.0
    c.loss.gradnorm_weight_cap = 10000.0
    c.loss.gradnorm_bias = (1.0, 1.0, 1.0, 1.0)  # [ac, ch, bc, ic]
    c.loss.bias_schedule = ()
    # Example: ((1000, 0.5, 2.0, 1000.0, 10000.0), (3000, 0.01, 1.0, 10.0, 10.0))
    # First value in each tuple is end_step, rest is the bias for that phase.
    # After the last phase, the bias fallbacks to the `gradnorm_bias` value.
    c.loss.use_causal = False
    c.loss.stagger_steps = 0       # stagger AC/CH for first N iters, 0=off
    c.loss.stagger_switch_every = 10
    c.loss.iface_gamma = 0.0   # interface weighting: w = 1+γ·4φ(1-φ), 0=off
    # Causal params
    c.causal = config_dict.ConfigDict()
    c.causal.n_chunks = 20
    c.causal.eps_init = 1.0
    c.causal.step_size = 1.2
    c.causal.max_eps = 10.0
    c.causal.max_last_weight = 0.99
    c.causal.min_mean_weight = 0.5

    # Logging
    c.log = config_dict.ConfigDict()
    c.log.log_dir = "/root/autodl-tmp/tf-logs/corrosion1d/"
    c.log.run_name = ""
    c.log.run_diag = False
    c.log.log_every = 20
    c.log.eval_every = 100
    c.log.save_ckpt = True

    # FEM reference
    c.fem = config_dict.ConfigDict()
    c.fem.data_path = "/root/autodl-tmp/data/corrosion1d/results_diffusion_cn.npz"

    # Initial condition
    c.init = config_dict.ConfigDict()
    c.init.center = 0.8

    # Physical — diffusion-controlled (L=2.0)
    c.physical = config_dict.ConfigDict()
    c.physical.D = 8.5e-10
    c.physical.A = 5.35e7
    c.physical.L = 2.0
    c.physical.alpha_phi = 1.03e-4
    c.physical.omega_phi = 1.76e7
    c.physical.c_se = 1.0
    c.physical.c_le = 5100.0 / 1.43e5

    # Non-dimensionalization
    c.nondim = config_dict.ConfigDict()
    c.nondim.L_c = 100e-6
    c.nondim.T_c = 50.0
    
    return c
