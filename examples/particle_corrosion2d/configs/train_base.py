"""Explicit first geometry-conditioned corrosion experiment; not yet SOTA."""
from ml_collections import ConfigDict
from examples.particle_corrosion2d.configs.geometry_base import get_config as geometry_config


def get_config():
    c = geometry_config()
    c.physical = ConfigDict(dict(kappa=2.0, n_ch=80.0,
        n_ac1=4.0*2.0*5.35e7/1.76e7, n_ac2=4.0,
        n_ac3=4.0/(2.0*30.0**2), c_se=1.0, c_le=5100.0/1.43e5))
    c.mesh = ConfigDict(dict(spacing=0.015, boundary_points=128))
    c.nl = ConfigDict(dict(method='nonlocal', delta=0.026,
        visibility_points=128, visibility_tolerance=1e-10,
        max_neighbors=128, max_halo=4096))
    c.sample = ConfigDict(dict(t_max=1.0, Nt=8, K_core=256, K_pool=8))
    c.operator = ConfigDict(dict(condition_batch=4, geometry_count=64,
        domain_shape=(25,41), wall_points_per_particle=64,
        wall_points_per_side=32, mouth_points=16, ic_points=1024))
    c.arch = ConfigDict(dict(input_dims=(3,5), extent=(1.0,0.6), width=96,
        heads=4, condition_depth=2, decoder_depth=3, spatial_features=48,
        spatial_scale=2.0, time_features=8, time_max_frequency=4.0,
        output_scale=1.0, norm_eps=1e-5))
    c.train = ConfigDict(dict(iterations=10000, optimizer='adam', lr_init=1e-3,
        lr_min=1e-5, lr_decay_every=2000, lr_decay_rate=0.5, grad_clip=10.0,
        alpha_gradnorm=0.9, gradnorm_update_every=25, compute_dtype='float32',
        matmul_precision='highest', gpu_policy='ordinary', soap_b1=0.95,
        soap_b2=0.9, soap_precond_freq=10, soap_max_precond_dim=256,
        soap_precond_1d=False))
    c.loss = ConfigDict(dict(ac_scale=1.0, ch_scale=1.0,
        gradnorm_bias=(1.0,1.0,1.0,1.0,1.0), gradnorm_weight_cap=1000.0,
        neumann_scales=(1.0,1.0)))
    c.reference = ConfigDict(dict(snapshots=11, conditions=4,
        rtol=1e-7, atol=1e-9, max_step=0.025,
        refined_spacing=0.01, refined_boundary_points=256,
        refined_rtol=2e-8, refined_atol=2e-10, refined_max_step=0.0125))
    c.log = ConfigDict(dict(log_dir='/root/autodl-tmp/tf-logs/particle_corrosion2d',
        log_every=100, eval_every=500, checkpoint_every=2500))
    return c
