"""Dimensionless periodic pilot; not an exact PFHub4 reproduction."""
from ml_collections import ConfigDict


def get_config():
    c = ConfigDict()
    c.geometry = ConfigDict(dict(length=1.))
    c.physical = ConfigDict(dict(beta=1., epsilon=.025, mobility=.01,
        lame_lambda=2450., shear_modulus=1225., eigenstrain=(.01, 0., 0.), terminal_time=1.))
    c.initial = ConfigDict(dict(center=(.5, .5), level=.35, width=.09, background=-.95))
    c.sample = ConfigDict(dict(Nx=63, Ny=63, Nt=8))
    c.nl = ConfigDict(dict(method='nonlocal', delta=.04))
    c.arch = ConfigDict(dict(hidden_dim=64, num_layers=4, spatial_modes=3,
        activation='gelu', temporal_emb_dim=8, temporal_scale=1.))
    c.train = ConfigDict(dict(iterations=5000, lr_init=1e-3, lr_decay_every=1000,
        lr_decay_rate=.95, lr_min=1e-5, optimizer='soap', grad_clip=10.,
        soap_b1=.9, soap_b2=.99, soap_precond_freq=5, soap_max_precond_dim=128,
        soap_precond_1d=False, compute_dtype='float64', gpu_policy='ordinary'))
    c.train.refinement = ConfigDict(dict(iterations=500, time_points=16,
        max_evaluations=1000, max_line_search=30, history_size=50,
        gradient_tolerance=1e-9, function_tolerance=1e-14))
    c.loss = ConfigDict(dict(phase_weight=1., chemical_weight=1.))
    c.reference = ConfigDict(dict(path='/root/autodl-tmp/data/precipitate2d/results_periodic_smallstrain_pilot.npz',
        local_grid=127, refined_local_grid=255, dt=.0005, refined_dt=.00025,
        snapshots=21, error_tolerance=.002))
    c.log = ConfigDict(dict(log_dir='/root/autodl-tmp/tf-logs/precipitate2d',
        run_name='', eval_every=250, checkpoint_every=1000))
    return c
