"""Larger 2D Swish/Fourier candidate inspired by fracture1d's damage network."""

from examples.fracture2d.configs.hybrid_pinn_short_damage import get_config as base_config


def get_config():
    c = base_config()
    c.arch.hidden_dim = 256
    c.arch.num_layers = 6
    c.arch.activation = "swish"
    c.arch.use_spatial_emb = True
    c.arch.spatial_emb_dim = 64
    c.arch.spatial_scale = 16.0
    # User-selected time-saving path: fresh solve at .0040, then du=.0001.
    # end_state_index counts selected-load ordinals, not reference indices.
    c.data.reference_indices = list(range(40, 76))
    c.data.end_state_index = 2  # Validate .0040/.0041 before extension to ordinal36.
    # The hybrid optimizer trains all inexact network leaves, including the
    # Fourier kernel. No frozen-frequency flag or 1D hat basis is imported.
    c.output.dir = "/root/autodl-tmp/tf-logs/fracture2d/hybrid_swish_fourier_256x6_first2_20261010"
    c.output.event_dir = "/root/autodl-tmp/tf-logs/fracture2d/events/hybrid_swish_fourier_256x6_first2_20261010"
    return c
