"""3D Ice Melting PINN training config."""

from ml_collections import config_dict


def get_config():
    c = config_dict.ConfigDict()

    # Architecture
    c.arch = config_dict.ConfigDict()
    c.arch.type = "mlp"               # "mlp", "modified_mlp", or "film"
    c.arch.hidden_dim = 64
    c.arch.film_hidden_dim = 64
    c.arch.film_num_layers = 1
    c.arch.num_layers = 4
    c.arch.spatial_emb_dim = 64
    c.arch.spatial_scale = 2.0 
    c.arch.temporal_emb_dim = 64
    c.arch.temporal_scale = 2.0
    c.arch.activation = "gelu"
    c.arch.use_spatial_emb = True
    c.arch.use_temporal_emb = True
    c.arch.out_dim = 1                 # single field φ
    c.arch.spatial_dim = 3             # 3D
    c.arch.use_hard_constraint = False  # tanh output

    # Non-local operator
    c.nl = config_dict.ConfigDict()
    c.nl.delta = 0.01                  # δ/h≈3.8, k_avg≈242, rel-L2≈7%
    c.nl.kernel = "cubic"
    c.nl.method = "sparse"

    # Sampling
    c.sample = config_dict.ConfigDict()
    c.sample.Nx = 128
    c.sample.Ny = 128
    c.sample.Nz = 128
    c.sample.Nt = 50                   # temporal batch size
    c.sample.t_min = 0.001             # avoid t=0 for PDE (IC handled separately)
    c.sample.K_core = 2000             # spatial minibatch core
    c.sample.K_pool = 200               # pre-built sub-graph pool entries
    c.sample.N_ic = 20000              # IC sample points

    # Training
    c.train = config_dict.ConfigDict()
    c.train.iterations = 50000
    c.train.lr_init = 5e-4
    c.train.lr_decay_every = 1000
    c.train.lr_decay_rate = 0.95
    c.train.lr_min = 1e-5
    c.train.optimizer = "adam"
    c.train.grad_clip = 100.0
    c.train.soap_max_precond_dim = 100000
    c.train.alpha_gradnorm = 1.0
    c.train.gradnorm_update_every = 20

    # Loss
    c.loss = config_dict.ConfigDict()
    c.loss.gradnorm_weight_cap = 1000.0
    c.loss.gradnorm_bias = (1.0, 1.0)  # [pde, ic]
    c.loss.iface_gamma_init = 10.0          # initial interface weight
    c.loss.iface_gamma_decay_steps = 1000  # linear decay to 0 over N steps
    c.loss.iface_gamma_decay_to = 0.0      # final value after decay
    c.loss.use_causal = False              # enable per-t causal weighting

    # Causal (per-t mode, n_chunks=None)
    c.causal = config_dict.ConfigDict()
    c.causal.eps_init = 1.0
    c.causal.max_eps = 100.0
    c.causal.step_size = 1.1
    c.causal.max_last_weight = 0.9
    c.causal.min_mean_weight = 0.01

    # Logging
    c.log = config_dict.ConfigDict()
    c.log.log_dir = "/root/autodl-tmp/tf-logs/icemelting3d/"
    c.log.run_name = ""
    c.log.log_every = 20
    c.log.eval_every = 200
    c.log.save_ckpt = True
    c.log.run_diag = False

    # Physical (dimensional + non-dimensional)
    c.physical = config_dict.ConfigDict()
    c.physical.M = 0.1
    c.physical.lam = 5.0
    c.physical.R0 = 35.0
    c.physical.eps_star = 0.0225    # fixed to N=64 FEM value; do NOT derive from N

    # Non-dimensionalization
    c.nondim = config_dict.ConfigDict()
    c.nondim.L_c = 100.0
    c.nondim.T_c = 4.0
    c.nondim.xc = (0.5, 0.5, 0.5)

    return c
