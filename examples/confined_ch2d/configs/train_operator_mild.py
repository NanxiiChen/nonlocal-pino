"""Historical finite geometry pilot; fixed ambient IC and mild coarsening."""
from ml_collections import ConfigDict
from examples.confined_ch2d.configs.train_base import get_config as baseline


def get_config():
    c = baseline()
    c.geometry.mesh_size = .045
    c.geometry.boundary_outer = 96
    c.geometry.boundary_per_hole = 48
    c.physical.epsilon = .06
    c.physical.terminal_time = .2
    c.initial = ConfigDict(dict(kind='fixed_ambient', amplitude=.25,
        waves=((1., 1., 1., 0.), (2., -1., .4, .7))))
    c.nl.delta = .11
    c.arch = ConfigDict(dict(input_dim=5, extent=(1., 1.), width=48, heads=4,
        latent_tokens=0, encoder_depth=2, decoder_depth=3, spatial_features=24,
        spatial_scale=1., time_features=4, output_scale=.1, norm_eps=1e-5))
    c.operator = ConfigDict(dict(action='train', dataset_seed=20261029,
        reference_python='/root/miniconda3/envs/fenicsenv/bin/python',
        dataset_path='/root/autodl-tmp/data/confined_ch2d/operator_geometry_v1',
        train_conditions=16, test_conditions=8, condition_batch=4,
        center_min=((.28, .46), (.60, .36)), center_max=((.42, .61), (.74, .51)),
        radius_min=(.05, .04), radius_max=(.085, .075), clearance=.025,
        maximum_sampling_attempts=10000, metric='biharmonic_resolvent',
        local_validation_indices=(16, 17)))
    c.reference.snapshots = 5
    c.reference.dt = .0025
    c.reference.refined_dt = .00125
    c.reference.mesh_sizes = (.035, .025)
    c.reference.phase_tolerance = .01
    c.reference.chemical_tolerance = .05
    c.train.optimizer = 'adam'
    c.train.compute_dtype = 'float64'
    c.train.iterations = 5000
    c.sample.Nt = 4
    c.log.eval_every = 500
    c.log.checkpoint_every = 1000
    return c
