"""FP64 128x6 Swish/Fourier stagger trial starting fresh at load .0048."""

from examples.fracture2d.configs.hybrid_pinn_swish_fourier import get_config as base_config


def get_config():
    c = base_config()
    c.arch.hidden_dim = 128
    c.arch.num_layers = 6
    c.arch.activation = "swish"
    c.arch.use_spatial_emb = True
    c.arch.spatial_emb_dim = 64
    c.arch.spatial_scale = 16.0
    c.data.reference_indices = list(range(48, 76))
    c.data.end_state_index = 2  # .0048/.0049; full selected endpoint28 is .0075.
    c.output.dir = "/root/autodl-tmp/tf-logs/fracture2d/hybrid_swish_fourier_128x6_from0048_first2_20261010"
    c.output.event_dir = "/root/autodl-tmp/tf-logs/fracture2d/events/hybrid_swish_fourier_128x6_from0048_first2_20261010"
    return c
