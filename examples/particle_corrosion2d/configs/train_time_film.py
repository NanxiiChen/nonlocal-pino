"""Matched 20k temporal control: accepted spatial attention and decoder FiLM."""
from ml_collections import ConfigDict
from examples.particle_corrosion2d.configs.train_default import get_config as baseline


def get_config():
    c = baseline()
    c.arch.temporal_mode = 'decoder_film'
    c.monitor = ConfigDict(dict(expected_conditions=64, chunk_size=512,
        metric='equal_point_space_time', render_training_figures=False))
    c.log.eval_every = 500
    return c
