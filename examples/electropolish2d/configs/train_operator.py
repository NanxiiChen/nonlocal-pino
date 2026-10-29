"""Opt-in physics-informed random-interface operator pilot; no future labels."""
from ml_collections import ConfigDict
from examples.electropolish2d.configs.train_base import get_config as single_config


def get_config():
    c = single_config()
    del c.surface['seed']  # random interface identities belong to dataset provenance
    c.operator = ConfigDict()
    c.operator.concentration_map = 'linear_liquid'
    c.operator.image_shape = (1, 32, 64)
    c.arch = ConfigDict()
    c.arch.model = 'conditional_field'
    c.arch.local_input = 'initial'  # zero: diagnostic ablation; IC supervision stays intact
    c.arch.conditioning = 'cross_attention'  # alternative: film
    c.arch.patch_shape = (4, 8)
    c.arch.extent = (1.0, 0.5)
    c.arch.width = 64
    c.arch.encoder_depth = 2
    c.arch.heads = 4
    c.arch.mlp_ratio = 2
    c.arch.decoder_depth = 3
    c.arch.spatial_features = 32
    c.arch.spatial_scale = 2.0
    c.arch.time_features = 8
    c.arch.time_scale = 1.0
    c.arch.norm_eps = 1e-5
    c.operator.train_conditions = 16
    c.operator.test_conditions = 4
    c.operator.eval_train_conditions = 2
    c.operator.condition_batch = 2
    c.operator.ic_points = 1024
    c.operator.dataset_path = '/root/autodl-tmp/data/electropolish2d/operator_random16_v1'
    c.operator.reference_action = 'load'  # prepare: CPU-only condition/reference preparation
    c.sample.Nt = 4
    c.sample.K_core = 256
    c.sample.K_pool = 32
    c.train.iterations = 2000
    c.train.optimizer = 'adam'  # inexpensive architecture pilot; SOAP remains explicit
    c.train.grad_clip = 10.0
    c.log.eval_every = 250
    c.log.log_every = 100
    c.log.run_name = ''
    return c
