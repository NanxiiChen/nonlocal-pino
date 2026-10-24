"""Optimizer factory: Adam and SOAP (upstream SOAP_JAX 0.2.2)."""

import jax, jax.numpy as jnp, optax
from soap_jax import soap


def make_optimizer(config):
    """Build optimizer from config.  config.train.optimizer = 'adam' | 'soap'."""
    opt_name = config.train.optimizer
    if opt_name == 'rprop':
        inner = optax.rprop(
            learning_rate=config.train.rprop_initial_step_size,
            eta_minus=config.train.rprop_eta_minus,
            eta_plus=config.train.rprop_eta_plus,
            min_step_size=config.train.rprop_min_step_size,
            max_step_size=config.train.rprop_max_step_size,
        )
        return optax.chain(optax.clip_by_global_norm(config.train.grad_clip), inner)
    schedule = optax.exponential_decay(
        init_value=config.train.lr_init,
        transition_steps=config.train.lr_decay_every,
        decay_rate=config.train.lr_decay_rate,
        end_value=config.train.lr_min,
    )
    if opt_name == 'adam':
        inner = optax.adam(schedule)
    elif opt_name == 'soap':
        inner = soap(
            learning_rate=schedule,
            b1=config.train.soap_b1,
            b2=config.train.soap_b2,
            precondition_frequency=config.train.soap_precond_freq,
            max_precond_dim=config.train.soap_max_precond_dim,
            precondition_1d=config.train.soap_precond_1d,
        )
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")
    return optax.chain(optax.clip_by_global_norm(config.train.grad_clip), inner)
