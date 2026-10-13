"""Fresh crack-biased random cores; compare resampling every8 versus32 updates."""
from examples.fracture2d.configs.core_halo_damage_trial import get_config as base


def get_config():
    c = base()
    c.core.output_dir = '/root/autodl-tmp/tf-logs/fracture2d/core_halo_damage_resample_20261012'
    c.core.cases = ['sample_crack_8', 'sample_crack_32']
    c.core.block_iterations = {'sample_crack_8':8, 'sample_crack_32':32}
    c.core.sampling_profile = 'geometric_inverse_power'
    c.core.boost = 49.
    c.core.density_power = 2.
    c.core.halo_capacity = 32768
    return c
