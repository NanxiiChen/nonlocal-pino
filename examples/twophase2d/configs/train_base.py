"""Small periodic advective-CH pilot; no claim of a tuned SOTA configuration."""
from ml_collections import ConfigDict


def get_config():
    c = ConfigDict()
    c.geometry = ConfigDict(dict(length=1.0))
    c.physical = ConfigDict(dict(epsilon=0.05, mobility=0.002,
                                velocity=(0.2, 0.1), terminal_time=1.0))
    c.initial = ConfigDict(dict(center=(0.35, 0.5), level=0.3, width=0.12))
    c.sample = ConfigDict(dict(Nx=32, Ny=32, Nt=4))
    c.nl = ConfigDict(dict(method='nonlocal', delta=0.08, kernel='cubic',
                          normalization='discrete_second_moment'))
    c.arch = ConfigDict(dict(hidden_dim=48, num_layers=3, activation='gelu',
                            spatial_modes=3, temporal_emb_dim=8, temporal_scale=1.0))
    c.train = ConfigDict(dict(iterations=5000, lr_init=1e-3, lr_decay_every=1000,
        lr_decay_rate=0.95, lr_min=1e-5, optimizer='soap', grad_clip=10.,
        soap_b1=0.9, soap_b2=0.99, soap_precond_freq=5, soap_max_precond_dim=128,
        soap_precond_1d=False, compute_dtype='float32', gpu_policy='ordinary',
        matmul_precision='highest'))
    c.loss = ConfigDict(dict(phase_weight=1., chemical_weight=1., mass_weight=1.))
    c.reference = ConfigDict(dict(
        path='/root/autodl-tmp/data/twophase2d/results_periodic_pilot.npz',
        local_grid=64, refined_local_grid=128, dt=0.0002, refined_dt=0.0001,
        snapshots=21, error_tolerance=0.002))
    c.log = ConfigDict(dict(log_dir='/root/autodl-tmp/tf-logs/twophase2d',
        run_name='', eval_every=200, checkpoint_every=1000))
    return c
