"""Initial nonlocal Neumann pilot; no SOTA or local-limit accuracy claim."""
from ml_collections import ConfigDict


def get_config():
    c = ConfigDict()
    c.geometry = ConfigDict(dict(center=(.5,.5), axes=(.5,.4),
        holes=((.35,.54,.075),(.68,.43,.055)), mesh_size=.025,
        boundary_outer=192, boundary_per_hole=64))
    c.physical = ConfigDict(dict(beta=1.,epsilon=.05,mobility=.02,terminal_time=1.))
    c.initial = ConfigDict(dict(amplitude=.3,wall_width=.04))
    c.nl = ConfigDict(dict(method='nonlocal',delta=.07,kernel='quadratic_visible'))
    c.sample = ConfigDict(dict(Nt=8,include_initial_time=True))
    c.arch = ConfigDict(dict(hidden_dim=64,num_layers=4,spatial_modes=3,
        activation='gelu',temporal_emb_dim=8,temporal_scale=1.))
    c.train = ConfigDict(dict(iterations=5000,lr_init=1e-3,lr_decay_every=1000,
        lr_decay_rate=.95,lr_min=1e-5,optimizer='soap',grad_clip=10.,soap_b1=.9,
        soap_b2=.99,soap_precond_freq=5,soap_max_precond_dim=128,
        soap_precond_1d=False,compute_dtype='float64',gpu_policy='ordinary'))
    c.train.refinement = ConfigDict(dict(iterations=500,time_points=16,
        max_evaluations=1000,max_line_search=30,history_size=50,
        gradient_tolerance=1e-9,function_tolerance=1e-14))
    c.loss = ConfigDict(dict(phase_weight=1.,chemical_weight=1.,
        boundary_phi_weight=1.,boundary_mu_weight=1.))
    c.reference = ConfigDict(dict(root='/root/autodl-tmp/data/confined_ch2d',
        path='/root/autodl-tmp/data/confined_ch2d/reference.npz',
        mesh_sizes=(.025,.0175),dt=.005,refined_dt=.0025,snapshots=21,
        phase_tolerance=.01,chemical_tolerance=.04))
    c.log = ConfigDict(dict(log_dir='/root/autodl-tmp/tf-logs/confined_ch2d',
        run_name='',eval_every=250,checkpoint_every=1000))
    return c
