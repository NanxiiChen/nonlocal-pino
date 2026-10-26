"""Canonical entry adapter for matched KKS training and explicit local controls."""
from nonlocal_pinn.training.runtime import configure_gpu_policy


def validate_matched_options(config):
    """Reject legacy-loop switches the matched trainer cannot honor."""
    if config.nl.method != "sparse" or config.sample.K_core <= 0:
        raise ValueError("Matched training requires a sparse graph and K_core>0; "
                         "use a local research preset or a legacy full-grid config")
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
    configure_gpu_policy(config.train.gpu_policy)
    from examples.corrosion2d.residual_experiment import train
    return train(config)
