"""Explicit single-state full-graph PINN experiment on the adopted 2D model."""

from ml_collections import ConfigDict


def get_config():
    c = ConfigDict()
    c.data = ConfigDict()
    c.data.cloud = "/root/autodl-tmp/data/fracture2d/engineering_grip/source/material_cloud_d048.npz"
    c.data.reference = "/root/autodl-tmp/data/fracture2d/engineering_grip/direct_nonlocal_final/solution_u0p0075_fresh_v1.npz"
    c.data.cloud_sha256 = "fa2ebf165f2e89ec88c63116346401b5039263e8e4c1a1917f6f13a3f4541267"
    c.data.reference_sha256 = "5f3bc12bc4bc31718cce4c4aed18d47314fd80e3a207bec0180ac6cdd93f7bab"
    c.data.state_index = 1
    c.data.end_state_index = 1
    c.nl = ConfigDict()
    c.nl.delta = 0.048
    c.nl.h_g = 0.02
    c.nl.visibility_tolerance = 1.0e-12
    c.physical = ConfigDict()
    c.physical.E = 210.0e3
    c.physical.nu = 0.3
    c.physical.k_res = 1.0e-8
    c.physical.rank_rtol = 1.0e-12
    c.physical.Gc = 2.7
    c.physical.l0 = 0.024
    # Smooth nonnegativity floor for the phase-field damage ansatz, in
    # physical damage units. Zero damage is the correct bulk value, so the
    # unconstrained least-squares output needs a one-sided floor.
    c.physical.damage_floor = 1.0e-8
    # Network architecture, consumed by `pinn.arch.make_pinn` through
    # `_build_network`. `use_spatial_emb` selects the built-in Fourier
    # embedding with `spatial_emb_dim` features per scale.
    c.arch = ConfigDict()
    c.arch.type = "mlp"
    c.arch.hidden_dim = 128
    c.arch.num_layers = 4
    c.arch.activation = "tanh"
    c.arch.use_spatial_emb = False
    c.arch.spatial_emb_dim = 32
    c.arch.spatial_scale = 16.0
    c.arch.multi_head = False
    c.arch.tensor_split = False
    c.arch.film_hidden_dim = 16
    c.arch.film_num_layers = 1
    c.train = ConfigDict()
    c.train.seed = 0
    c.train.dtype = "float32"
    c.train.steps = 100
    c.train.learning_rate = 1.0e-4
    c.train.report_every = 10
    c.loss = ConfigDict()
    # Optimization-only scales; raw residuals are always reported.
    c.loss.scaling = "initial_driver_rms"
    c.loss.mechanics_scale = 1.0
    c.loss.damage_scale = 100.0
    c.loss.history_scale = 1.0
    c.stagger = ConfigDict()
    c.stagger.sweeps = 1
    c.stagger.mechanics_steps = 100
    c.stagger.damage_steps = 100
    c.stagger.omega = 0.5
    # Acceptance gates for one completed sweep (reference-comparison based).
    c.stagger.mechanics_loss_limit = 1.0e-2
    c.stagger.displacement_l2_limit = 5.0e-2
    c.stagger.gate_displacement_l2 = True
    # Zero disables the reference-field RMSE gate while retaining diagnostics.
    c.stagger.damage_rmse_limit = 0.0
    c.stagger.reaction_relative_limit = 5.0e-2
    c.stagger.damage_range_atol = 1.0e-10
    c.stagger.healing_atol = 1.0e-3
    c.stagger.mechanics_scale_policy = "first_load_initial"
    c.stagger.accept_on_first_passing_report = True
    c.output = ConfigDict()
    # PINN artifacts and TensorBoard events both live under tf-logs, in
    # separate sibling directories so their identities never collide.
    c.output.dir = "/root/autodl-tmp/tf-logs/fracture2d/pinn_stagger_first_strict_fp32"
    c.output.tensorboard_root = "/root/autodl-tmp/tf-logs/fracture2d/events"
    c.output.log_every = 500
    c.output.checkpoint_every = 1000
    c.output.resume = False
    # Optional read-only accepted prefix for a new numerical-policy run.
    c.output.fork_from = ""
    c.output.fork_after_state_index = 0
    return c
