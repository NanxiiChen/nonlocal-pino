"""Invertible phase output coordinate and its exact transport residual metric."""
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np


def mapped_phase(initial,correction,spec):
    """Use a finite initial latent value; difference form preserves exact ICs."""
    latent_initial=jnp.arctanh(initial/spec.amplitude)
    latent=latent_initial+correction
    phase=initial+spec.amplitude*(jnp.tanh(latent)-jnp.tanh(latent_initial))
    return phase,latent


def map_jacobian(latent,amplitude):
    # Unlike 1-tanh(s)^2 this does not cancel to zero prematurely in FP32.
    return 4*amplitude*jax.nn.sigmoid(2*latent)*jax.nn.sigmoid(-2*latent)


def transformed_transport(residual,jacobian,amplitude,minimum_jacobian):
    """Exact, differentiable a/J scaling, failing rather than flooring J."""
    jacobian=eqx.error_if(jacobian,
        jnp.any(~jnp.isfinite(jacobian))|jnp.any(jacobian<minimum_jacobian),
        'Phase map Jacobian below explicit minimum or nonfinite; inverse-map residual is singular')
    return amplitude*residual/jacobian


def jacobian_diagnostics(jacobian,amplitude):
    """Host-only report; zero compute-dtype J is explicit, never floored."""
    jacobian=np.asarray(jacobian)
    if not np.isfinite(jacobian).all() or np.any(jacobian<0):
        raise FloatingPointError('Nonfinite or negative phase map Jacobian')
    minimum=float(jacobian.min())
    return dict(jacobian_min=minimum,jacobian_median=float(np.median(jacobian)),
        maximum_inverse_factor=float(amplitude)/minimum if minimum>0 else None,
        inverse_factor_status='finite' if minimum>0 else 'jacobian_underflow_in_compute_dtype')
