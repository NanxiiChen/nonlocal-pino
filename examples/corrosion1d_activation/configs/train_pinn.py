"""Training config for split-PINN on 1D KKS corrosion."""

from ml_collections import config_dict


def get_config():
    c = config_dict.ConfigDict()

    # Architecture
    c.arch = config_dict.ConfigDict()
    c.arch.rank_p = 50
    c.arch.hidden_dim = 64
    c.arch.num_layers = 4
    c.arch.spatial_emb_dim = 64
    c.arch.spatial_scale = 2.0
    c.arch.temporal_emb_dim = 64
    c.arch.temporal_scale = 2.0
    c.arch.activation = "gelu"

    # Non-local operator
    c.nl = config_dict.ConfigDict()
    c.nl.delta = 0.05
    c.nl.kernel = "cubic"  # "cubic" (smooth) or "tophat" (discontinuous)

    # Sampling
    c.sample = config_dict.ConfigDict()
    c.sample.Nx = 501
    c.sample.Nt = 100

    # Training
    c.train = config_dict.ConfigDict()
    c.train.iterations = 50000
    c.train.lr_init = 5e-4
    c.train.lr_decay_every = 500
    c.train.lr_decay_rate = 0.95
    c.train.lr_min = 1e-5
    c.train.optimizer = "adam"  # "adam" or "soap"
    c.train.grad_clip = 1.0
    # SOAP params (only used when optimizer == "soap")
    c.train.soap_b1 = 0.95
    c.train.soap_b2 = 0.95
    c.train.soap_precond_freq = 5
    c.train.soap_max_precond_dim = 10000
    c.train.soap_precond_1d = False  # small network: precondition all params
    c.train.log_every = 10
    c.train.eval_every = 100
    c.train.alpha_gradnorm = 1.0

    # Loss
    c.loss = config_dict.ConfigDict()
    c.loss.ac_scale = 1.0
    c.loss.ch_scale = 1/8500
    c.loss.gradnorm_weight_cap = 100000.0
    c.loss.weight_mode = "gradnorm"  # "gradnorm" or "fixed"
    c.loss.fixed_weights = [1.0, 1.0, 1.0, 0.01]  # [w_ac, w_ch, w_bc, w_ic]

    # Logging
    c.log = config_dict.ConfigDict()
    c.log.log_dir = "/root/autodl-tmp/tf-logs/corrosion1d_activation/"
    c.log.run_name = None

    # FEM reference
    c.fem = config_dict.ConfigDict()
    c.fem.data_path = "/root/autodl-tmp/data/corrosion1d_activation/results.npz"

    # Physical (same as FEM solver)
    c.physical = config_dict.ConfigDict()
    c.physical.D = 8.5e-10
    c.physical.A = 5.35e7
    c.physical.L = 1e-11
    c.physical.alpha_phi = 1.03e-4
    c.physical.omega_phi = 1.76e7
    c.physical.c_se = 1.0
    c.physical.c_le = 5100.0 / 1.43e5

    # Non-dimensionalization
    c.nondim = config_dict.ConfigDict()
    c.nondim.L_c = 100e-6
    c.nondim.T_c = 100000.0

    return c
