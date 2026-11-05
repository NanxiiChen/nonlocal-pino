"""Physics-only materials-map operator pilot; dimensionless time spans [0, 1]."""
from ml_collections import ConfigDict


def get_config():
    c = ConfigDict()
    c.geometry = ConfigDict(dict(length=1.0, height=0.6, pit_x=0.5,
        pit_radius=0.12, mouth_half_width=0.055, interface_k=30.0))
    # KKS chemical energy ratio 2A/w = 2*5.35e7/1.76e7, inherited from
    # corrosion2d. L0*w*T=4 and sqrt(w/(2alpha))*Lc=30 set the pilot units.
    # The independently chosen D*T/Lc**2=80 is checked by a D-doubling study:
    # area advance changes0.76% at kappa1 and3.32% at kappa3 on the local grid.
    c.physical = ConfigDict(dict(n_ch=80.0, n_ac1=4.0*2.0*5.35e7/1.76e7,
        n_ac2=4.0, n_ac3=4.0/(2.0*30.0**2), c_se=1.0,
        c_le=5100.0/1.43e5))
    c.material = ConfigDict(dict(minimum=1.0, maximum=3.0,
        modes=((1, 0), (0, 1), (1, 1), (2, 0), (0, 2), (2, 1)),
        coefficient_scale=0.85))
    c.sample = ConfigDict(dict(Nx=81, Ny=49, Nt=8, K_core=256, K_pool=32, t_max=1.0))
    c.nl = ConfigDict(dict(delta=0.026, kernel='cubic', method='sparse'))
    c.operator = ConfigDict(dict(image_shape=(1, 32, 64), condition_batch=8,
        ic_points=1024, reference_conditions=4, reference_snapshots=11))
    c.arch = ConfigDict(dict(width=64, encoder_depth=2, heads=4, mlp_ratio=2,
        decoder_depth=3, spatial_features=32, spatial_scale=2.0,
        time_features=8, time_scale=1.0, norm_eps=1e-5,
        patch_shape=(4, 8), extent=(1.0, 0.6), conditioning='cross_attention',
        local_dim=0))
    c.train = ConfigDict(dict(iterations=5000, optimizer='adam', lr_init=1e-3,
        lr_min=1e-5, lr_decay_every=1000, lr_decay_rate=0.5, grad_clip=10.0,
        alpha_gradnorm=0.9, gradnorm_update_every=25, compute_dtype='float32',
        matmul_precision='highest', gpu_policy='ordinary', soap_b1=0.95,
        soap_b2=0.9, soap_precond_freq=10, soap_max_precond_dim=256,
        soap_precond_1d=False))
    c.loss = ConfigDict(dict(ac_scale=1.0, ch_scale=1.0,
        gradnorm_bias=(1.0, 1.0, 1.0, 1.0, 1.0), gradnorm_weight_cap=1000.0))
    c.reference = ConfigDict(dict(rtol=1e-7, atol=1e-9, max_step=0.025,
        refined_rtol=2e-8, refined_atol=2e-10, refined_max_step=0.0125))
    c.log = ConfigDict(dict(log_dir='/root/autodl-tmp/tf-logs/heterogeneous_corrosion2d',
        run_name='', log_every=100, eval_every=250, checkpoint_every=1000))
    return c
