"""Validated 5 s SOAP default for the IC-only latent melting PINN.

Matches the 5000-update SOAP preset in docs/tasks/archive/icemelting-soap-adam-5s.md.
Selected checkpoint provenance is recorded in artifacts/index.yaml.
"""

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
    c.arch.temporal_scale = 1.0
    c.arch.activation = "tanh"
    c.arch.use_spatial_emb = True
    c.arch.use_temporal_emb = True
    c.arch.out_dim = 1                 # single field φ
    c.arch.spatial_dim = 3             # 3D
    c.arch.use_hard_constraint = False  # legacy output-only tanh; IC constraint is configured below
    # Split the first trunk layer into x-only + t-only matmuls (mlp trunk
    # only).  Exact same math — under vmap(x)xvmap(t) the dominant first-layer
    # matmul is hoisted out of the batch dimensions. Enabled with the validated
    # matmul_precision="highest" setting below.
    c.arch.tensor_split = True

    c.phase = config_dict.ConfigDict()
    c.phase.formulation = "ic_latent"  # "ic_latent" | "ic_phase" | "raw_phase"

    # Non-local operator
    c.nl = config_dict.ConfigDict()
    c.nl.delta = 0.01                  # 128 nodes/axis: delta/h=1.27, six axial neighbours
    c.nl.kernel = "cubic"
    c.nl.method = "sparse"             # "sparse" | "fd" | "local"
    # "local": autograd Laplacian via jax.hessian, LHS random sampling.
    #   No D matrix, no grid requirement.  Intended as ablation baseline
    #   to isolate the effect of the nonlocal operator.  ~5-10× slower
    #   per step than sparse/fd due to per-point Hessian computation.

    # Sampling
    c.sample = config_dict.ConfigDict()
    c.sample.Nx = 128
    c.sample.Ny = 128
    c.sample.Nz = 128
    c.sample.Nt = 64                   # temporal batch size
    c.sample.t_min = 0.001             # avoid t=0 for PDE (IC handled separately)
    c.sample.K_core = 2000             # spatial minibatch core
    c.sample.K_pool = 200               # pre-built sub-graph pool entries
    c.sample.N_lhs = 2000               # LHS points per step (local mode only)
    c.sample.N_ic = 20000              # IC sample points

    # Training
    c.train = config_dict.ConfigDict()
    c.train.iterations = 5000
    c.train.lr_init = 1e-3
    c.train.lr_decay_every = 1000
    c.train.lr_decay_rate = 0.95
    c.train.lr_min = 1e-5
    c.train.optimizer = "soap"
    c.train.grad_clip = 100.0
    # "" keeps JAX's default matmul precision (this script's historical
    # behavior).  Set "highest" for exact FP32 or "tensorfloat32" explicitly.
    c.train.matmul_precision = "highest"
    c.train.alpha_gradnorm = 0.9
    c.train.gradnorm_update_every = 1
    c.train.soap_b1 = 0.90
    c.train.soap_b2 = 0.99
    c.train.soap_precond_freq = 5
    c.train.soap_max_precond_dim = 100000
    c.train.soap_precond_1d = False


    # Loss
    c.loss = config_dict.ConfigDict()
    c.loss.gradnorm_weight_cap = 1000.0
    c.loss.gradnorm_bias = (1.0, 1.0)  # [pde, ic]
    c.loss.iface_gamma_init = 0.0          # initial interface weight
    c.loss.iface_gamma_decay_steps = 0  # linear decay to 0 over N steps
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

    c.eval = config_dict.ConfigDict()
    c.eval.graph_reference_path = ""   # optional verified accuracy.py reference.npz, evaluation only
    c.eval.full_grid_every = 1000      # terminal spatial L2; also at initialization/final step
    c.eval.chunk_size = 32768          # fixed-shape full-grid inference

    # Physical (dimensional + non-dimensional)
    c.physical = config_dict.ConfigDict()
    c.physical.M = 0.1
    c.physical.lam = 5.0
    c.physical.R0 = 35.0
    c.physical.eps_star = 0.0225    # fixed to N=64 FEM value; do NOT derive from N

    # Non-dimensionalization
    c.nondim = config_dict.ConfigDict()
    c.nondim.L_c = 100.0
    c.nondim.T_c = 5.0
    c.nondim.xc = (0.5, 0.5, 0.5)

    return c
