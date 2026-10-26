"""Run provenance independent of a numerical strategy's configuration."""
from numbers import Integral
import os
import shlex

DEFAULT_RUN_SEED = 42


def configure_gpu_policy(policy):
    """Set an explicit compilation policy before backend initialization.

    Preserve unrelated flags and reject a conflicting inherited policy rather
    than silently running a different experiment. No JAX backend is imported.
    """
    if policy not in ("deterministic", "ordinary"):
        raise ValueError(f"Unknown train.gpu_policy: {policy}")
    flags = os.environ["XLA_FLAGS"] if "XLA_FLAGS" in os.environ else ""
    settings = [flag for flag in shlex.split(flags)
                if flag == "--xla_gpu_deterministic_ops"
                or flag.startswith("--xla_gpu_deterministic_ops=")]
    if any(flag not in ("--xla_gpu_deterministic_ops", "--xla_gpu_deterministic_ops=true")
           for flag in settings):
        raise ValueError("Remove the conflicting XLA_FLAGS deterministic setting; "
                         "select the policy with train.gpu_policy")
    if policy == "ordinary" and settings:
        raise ValueError("Ordinary GPU policy requires no deterministic flag in XLA_FLAGS")
    if not settings and policy == "deterministic":
        flags = (flags + " --xla_gpu_deterministic_ops=true").strip()
    os.environ["XLA_FLAGS"] = flags
    return flags


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
