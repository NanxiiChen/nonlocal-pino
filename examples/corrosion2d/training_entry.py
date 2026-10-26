"""Canonical entry adapter for the validated matched-graph KKS trainer."""
import os
import shlex


def validate_matched_options(config):
    """Reject legacy-loop switches the matched trainer cannot honor."""
    if config.nl.method != "sparse" or config.sample.K_core <= 0:
        raise ValueError("Matched training requires a sparse graph and K_core>0; "
                         "select a legacy config for FD/local/full-grid controls")
    if (config.loss.use_causal or config.loss.iface_gamma_init != 0
            or config.loss.iface_gamma_decay_steps != 0
            or config.loss.iface_gamma_decay_to != 0 or config.loss.bias_schedule):
        raise ValueError("Matched training does not support causal/interface weighting "
                         "or bias schedules; select a legacy config for those controls")
    if not config.log.save_ckpt or config.log.run_diag:
        raise ValueError("Matched training requires save_ckpt=True and run_diag=False; "
                         "its field/transport diagnostics are always recorded")


def train_matched(config, case):
    """Dispatch a fully explicit config without importing another CLI module."""
    if config.experiment.case != case:
        raise ValueError(f"{case}/train.py received config for {config.experiment.case}")
    validate_matched_options(config)
    # Set the audited GPU compilation setting before JAX backend initialization.
    flags = os.environ["XLA_FLAGS"] if "XLA_FLAGS" in os.environ else ""
    settings = [flag for flag in shlex.split(flags)
                if flag.startswith("--xla_gpu_deterministic_ops=")]
    if any(flag != "--xla_gpu_deterministic_ops=true" for flag in settings):
        raise ValueError("Matched training requires --xla_gpu_deterministic_ops=true; "
                         "remove the conflicting XLA_FLAGS setting")
    if not settings:
        os.environ["XLA_FLAGS"] = (flags+" --xla_gpu_deterministic_ops=true").strip()
    from examples.corrosion2d.residual_experiment import train
    return train(config)
