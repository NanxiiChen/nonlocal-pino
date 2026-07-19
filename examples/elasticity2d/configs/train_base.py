"""2D Elastic Plate with Hole — PINN training config."""

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
    c.arch.activation = "gelu"
    c.arch.use_spatial_emb = True
    c.arch.use_temporal_emb = False   # steady-state
    c.arch.out_dim = 2                # (u_x, u_y)
    c.arch.spatial_dim = 2
    c.arch.use_hard_constraint = True  # left clamped, right pulled
    c.arch.tensor_split = False
    c.arch.multi_head = True           # separate output heads per component

    # Non-local operator
    c.nl = config_dict.ConfigDict()
    c.nl.delta = 0.01                 # horizon
    c.nl.kernel = "cubic"
    c.nl.method = "sparse"            # "sparse" | "local"

    # Spatial sampling (regular grid with hole excluded)
    c.sample = config_dict.ConfigDict()
    c.sample.Nx = 200                 # grid points in x
    c.sample.Ny = 150                 # grid points in y
    c.sample.K_core = 0               # 0 = full batch (no minibatch)
    c.sample.K_pool = 0

    # Training
    c.train = config_dict.ConfigDict()
    c.train.iterations = 20000
    c.train.lr_init = 5e-4
    c.train.lr_decay_every = 500
    c.train.lr_decay_rate = 0.95
    c.train.lr_min = 1e-5
    c.train.optimizer = "adam"
    c.train.grad_clip = 100.0
    c.train.matmul_precision = "highest"
    c.train.soap_max_precond_dim = 100000
    c.train.soap_b1 = 0.90
    c.train.soap_b2 = 0.99
    c.train.soap_precond_freq = 5
    c.train.soap_precond_1d = False

    # Loss
    c.loss = config_dict.ConfigDict()
    c.loss.gradnorm_weight_cap = 1000.0
    c.loss.gradnorm_bias = (1.0,)     # single term

    # Logging
    c.log = config_dict.ConfigDict()
    c.log.log_dir = "/root/autodl-tmp/tf-logs/elasticity2d/"
    c.log.run_name = ""
    c.log.log_every = 20
    c.log.eval_every = 200
    c.log.save_ckpt = True

    # Physical (non-dimensional, plane strain, steel)
    # Steel: E=200 GPa, ν=0.3,  L_c=2 m, u_c=2 mm
    # Non-dim Lamé: depend only on ν
    c.physical = config_dict.ConfigDict()
    c.physical.E = 1.0
    c.physical.nu = 0.3
    c.physical.lam = 0.5769           # λ* = ν/((1+ν)(1-2ν))
    c.physical.mu = 0.3846            # μ* = 1/(2(1+ν))

    # Geometry + loading (non-dimensional)
    # Physical: 2m × 1.5m plate, 0.3m hole, 2mm displacement
    # L_c = 2m → Lx*=1.0, Ly*=0.75, a*=0.15, u_load*=1.0
    c.geometry = config_dict.ConfigDict()
    c.geometry.Lx = 1.0              # Lx* = 2m / 2m
    c.geometry.Ly = 0.75             # Ly* = 1.5m / 2m
    c.geometry.a = 0.15              # a* = 0.3m / 2m
    c.geometry.u_load = 1.0          # u_load* = 2mm / 2mm

    # FEM reference data
    c.fem = config_dict.ConfigDict()
    c.fem.data_path = "/root/autodl-tmp/data/elasticity2d/results.npz"

    return c
