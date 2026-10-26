"""Run provenance independent of a numerical strategy's configuration."""
from numbers import Integral

DEFAULT_RUN_SEED = 42


def with_run_seed(config, seed):
    """Resolve CLI seed, retaining serialized seeds when loading old configs.

    Canonical strategy configs omit train.seed. The resolved copy records it
    for reproducibility and old checkpoint readers; the input is not mutated.
    """
    if seed is None:
        seed = config.train.seed if 'seed' in config.train else DEFAULT_RUN_SEED
    if isinstance(seed, bool) or not isinstance(seed, Integral) or not 0 <= seed < 2**32:
        raise ValueError('Run seed must be an integer in [0, 2**32)')
    resolved = config.copy_and_resolve_references()
    with resolved.train.unlocked():
        resolved.train.seed = int(seed)
    return resolved
