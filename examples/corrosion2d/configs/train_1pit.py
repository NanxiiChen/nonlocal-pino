"""Training config for full-MLP PINN on 2D KKS corrosion (diffusion-controlled)."""

from ml_collections import config_dict


def get_config():
    c = config_dict.ConfigDict()

    # Architecture
    c.arch = config_dict.ConfigDict()
    c.arch.type = "mlp"  # "mlp" or "modified_mlp"
    c.arch.hidden_dim = 64
    c.arch.num_layers = 4
    c.arch.spatial_emb_dim = 64
    c.arch.spatial_scale = 2.0
    c.arch.temporal_emb_dim = 64
    c.arch.temporal_scale = 1.0
    c.arch.activation = "gelu"
    c.arch.use_spatial_emb = True
    c.arch.use_temporal_emb = True
    c.arch.use_hard_constraint = True

    # Non-local operator
    c.nl = config_dict.ConfigDict()
    c.nl.delta = 0.02
    c.nl.kernel = "cubic"  # cubic is C¹ smooth, higher accuracy than tophat
    c.nl.method = "sparse"  # None=auto: dense if N<2000 else sparse

    # Sampling
    c.sample = config_dict.ConfigDict()
    c.sample.Nx = 200
    c.sample.Ny = 100
    c.sample.Nt = 50
    c.sample.t_min = 0.001
    c.sample.N_bc_dir = 1000     # Dirichlet BC: pit mouth (φ=0, c=0)
    c.sample.N_bc_solid = 500    # Dirichlet BC: left/right/top edges (φ=1, c=1)
    c.sample.N_ic_global = 2000  # IC: global uniform samples
    c.sample.N_ic_local = 2000   # IC: extra samples near pit interface

    # Pit geometry (matches FEM solver config)
    c.pits = config_dict.ConfigDict()
    c.pits.entries = ((0.5, 0.0, 0.1),)

    # Training
    c.train = config_dict.ConfigDict()
    c.train.iterations = 50000
    c.train.lr_init = 5e-4
    c.train.lr_decay_every = 1000
    c.train.lr_decay_rate = 0.95
    c.train.lr_min = 1e-5
    c.train.optimizer = "adam"
    c.train.grad_clip = 100.0
    c.train.alpha_gradnorm = 1.0
    c.train.gradnorm_update_every = 20  # update GradNorm weights every N iters; 1=every iter
    c.train.soap_b1 = 0.95
    c.train.soap_b2 = 0.95
    c.train.soap_precond_freq = 5
    c.train.soap_max_precond_dim = 10000
    c.train.soap_precond_1d = False

    # Loss
    c.loss = config_dict.ConfigDict()
    c.loss.ac_scale = 1e-7
    c.loss.ch_scale = 1.0
    c.loss.gradnorm_weight_cap = 10000.0
    c.loss.gradnorm_bias = (1.0, 1.0, 1.0, 1.0, 1.0)  # [ac, ch, bc_pit, bc_solid, ic]
    c.loss.iface_gamma_init = 5.0       # initial interface weight
    c.loss.iface_gamma_decay_steps = 4000   # decay to 0 over N steps; 0=disabled
    c.loss.iface_gamma_decay_to = 0.0    # final value after decay
    c.loss.bias_schedule = ()
    c.loss.use_causal = False
    c.loss.use_stagger = False
    c.loss.stagger_switch_every = 25
    # Causal params (used when use_causal=True)
    c.causal = config_dict.ConfigDict()
    c.causal.n_chunks = 10
    c.causal.eps_init = 1.0
    c.causal.step_size = 1.5
    c.causal.max_eps = 1000.0
    c.causal.max_last_weight = 0.99
    c.causal.min_mean_weight = 0.5

    # Logging
    c.log = config_dict.ConfigDict()
    c.log.log_dir = "/root/autodl-tmp/tf-logs/corrosion2d/"
    c.log.run_name = ""
    c.log.log_every = 20
    c.log.eval_every = 500
    c.log.save_ckpt = True
    c.log.run_diag = False  # residual breakdown (expensive: runs _compute_all twice)

    # FEM reference
    c.fem = config_dict.ConfigDict()
    c.fem.data_path = "/root/autodl-tmp/data/corrosion2d/results_1pit.npz"

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
