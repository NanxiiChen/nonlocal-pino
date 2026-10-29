"""Recommended online 3D melting NO: Adam B16, hard IC and latent PDE residual."""
from ml_collections import ConfigDict
from examples.icemelting3d.configs.train_base import get_config as base_config


def get_config():
    c = base_config()
    c.arch = ConfigDict(dict(type='conditional_field', image_shape=(1, 32, 32, 32),
        patch_shape=(4, 4, 4), extent=(1., 1., 1.), width=48, encoder_depth=2,
        heads=4, mlp_ratio=2, decoder_depth=3, spatial_features=32, spatial_scale=2.,
        time_features=8, time_scale=1., conditioning='cross_attention', norm_eps=1e-5))
    c.initial = ConfigDict(dict(kind='ellipsoid_sdf', center_min=.45, center_max=.55,
        axis_min=.24, axis_max=.36, sdf_iterations=40, reference_sdf_iterations=56))
    c.operator = ConfigDict(dict(condition_batch=16, action='train', probe_steps=8,
        dataset_path='/root/autodl-tmp/data/icemelting3d/operator_ellipsoids_v1',
        monitor_conditions=2, test_conditions=8, dataset_seed=20261029,
        reference_dt=.005, reference_tolerance=1e-5, snapshots=5,
        evaluation_axis_points=24, evaluation_chunk=1024))
    c.sample.Nt=8
    c.sample.K_core=256
    c.sample.K_pool=32
    c.train.optimizer='adam'
    c.train.iterations=5000
    c.log.eval_every=250
    c.log.log_every=50
    return c
