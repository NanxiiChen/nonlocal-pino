"""Own-zero L1/random Adam loading with an explicit coarse/fine load grid."""
from ml_collections import ConfigDict
from examples.fracture2d.configs.plain_sigmoid_cold import get_config as base


def get_config():
    previous = base()
    c = ConfigDict({key: previous[key].to_dict() for key in
        ('physical', 'geometry', 'nl', 'arch', 'train', 'data', 'trial',
         'conditioned', 'validation', 'outer_stop', 'reporting', 'stagger')})
    c.head = ConfigDict(dict(initial_damage=1e-3))
    c.outer_stop.max_sweeps = 128
    c.reporting.range_atol = 1e-3
    c.l1_loading = ConfigDict(dict(
        output_dir='/root/autodl-tmp/tf-logs/fracture2d/l1_damage_loading_from_zero_20261013',
        transition_load=.0050, coarse_step=5e-4, fine_step=1e-4, end_load=.0060,
        basis_path=previous.cold.basis_path, basis_sha256=previous.cold.basis_sha256,
        initial_physical_damage=0., initial_physical_history=0.,
        output_head='sigmoid', history_feature_scaling='physical',
        parameterization='latent_basis_rms', trunk_gain=1.,
        loss_norm='l1', explicit_residual_weighting=False,
        optimizer='adam', learning_rate=1e-5, decay_steps=512,
        decay_rate=1., minimum_rate=1e-5, grad_clip=1000.,
        b1=.9, b2=.999, eps=1e-8, eps_root=0., shampoo_beta=.95,
        correct_bias=True, precondition_frequency=5, max_precond_dim=128,
        precondition_1d=False, weight_decay=0.,
        inner_updates=512, sample_count=2048, halo_capacity=32768, pool_size=256,
        resample_every=1, seed=1729, warmup_seed=42,
        segment_start=[.5, .5], segment_end=[1., .5],
        sampling_width=.024, sampling_boost=9., sampling_power=2.,
        checkpoint_sweeps=[1, 8, 32, 64, 128], user_provisional=True,
        cpu_threads=4))
    return c
