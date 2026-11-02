"""Mild matched-property full NS--CH pilot; not an accuracy/SOTA claim."""
from ml_collections import ConfigDict


def get_config():
    c=ConfigDict()
    c.geometry=ConfigDict(dict(length=4.,height=1.,center=2.,half_length=.9,
        upper=.12,lower=.08,amplitude_min=.04,amplitude_max=.14))
    c.physical=ConfigDict(dict(density=1.,viscosity=.2,mean_velocity=.2,
        surface_tension=.2,epsilon=.05,mobility=.0005,terminal_time=6.,ramp_time=1.))
    c.initial=ConfigDict(dict(center=(.8,.5),radius=.22,transition=.05))
    c.lifting=ConfigDict(dict(kind='normalized',time_scale=.25,inlet_length_scale=.05))
    c.velocity=ConfigDict(dict(kind='streamfunction'))
    c.sample=ConfigDict(dict(Nx=96,Ny=40,Nt=4,boundary_points=48,include_initial_time=False))
    c.arch=ConfigDict(dict(width=96,heads=4,condition_depth=2,decoder_depth=3,
        spatial_features=48,spatial_scale=2.,time_features=8,time_max_frequency=2.,
        output_scale=1.,norm_eps=1e-5))
    c.train=ConfigDict(dict(iterations=5000,lr_init=1e-3,lr_decay_every=2000,
        lr_decay_rate=.7,lr_min=1e-5,optimizer='adam',grad_clip=1.,soap_b1=.9,
        soap_b2=.9,soap_precond_freq=5,soap_max_precond_dim=256,
        soap_precond_1d=False,compute_dtype='float32',matmul_precision='highest',
        gpu_policy='ordinary'))
    c.operator=ConfigDict(dict(train_conditions=16,condition_batch=2,
        dataset_seed=20261031))
    c.nl=ConfigDict(dict(method='gmls',neighbors=36,degree=3,weight_exponent=16.,
        condition_limit=1e9))
    c.loss=ConfigDict(dict(momentum_scale=1.,phase_scale=.2,chemical_scale=4.,
        continuity_scale=.2,boundary_mu_scale=4.,boundary_phi_scale=1.,
        traction_scale=1.,pressure_scale=1.,stream_scale=.2))
    c.reference=ConfigDict(dict(root='/root/autodl-tmp/data/channel_nsch2d',
        coarse_grid=(40,16),fine_grid=(64,24),dt=.05,refined_dt=.025,snapshots=13,
        phase_tolerance=.025,velocity_tolerance=.025,pressure_tolerance=.08,
        linear_solver='mumps',newton_tolerance=1e-9,newton_iterations=25,time_scheme='bdf2'))
    c.log=ConfigDict(dict(log_dir='/root/autodl-tmp/tf-logs/channel_nsch2d',
        run_name='',eval_every=250,checkpoint_every=2500))
    return c
