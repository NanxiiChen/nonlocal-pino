"""Mixed advective CH; nonlocal diffusion with AD first spatial derivatives."""
import jax
import jax.numpy as jnp
from examples.twophase2d.problem import velocity_field
from examples.twophase2d.fields import predict_fields
from examples.twophase2d.advection import make_flux_divergence
from examples.twophase2d.preconditioning import make_schur_transform, make_causal_transform


def inverse_jacobian_weight(phi, eta):
    """Bounded inverse tanh-J continuation; full AD, no output clipping.

    Smooth near phi=+-1. The explicit denominator cap gives w>=1, preventing
    large excursions from reducing residual weights. This is a preconditioner,
    not an exact tanh output representation or an interface indicator.
    """
    return jnp.sqrt(1+eta**2)/jnp.sqrt(eta**2+jnp.minimum((1-phi**2)**2, 1.))


def make_residual_fn(x_all, D, config):
    """Original physical mixed residuals, available for independent loss audits."""
    epsilon2 = config.physical.epsilon**2
    mobility = config.physical.mobility
    flux_divergence = (make_flux_divergence(config, dtype=x_all.dtype)
                       if 'advection' in config else None)

    def residuals(net, t):
        fields, time_derivative = jax.jvp(
            lambda tt: predict_fields(net, x_all, tt, config),
            (t,), (jnp.ones_like(t),))
        if D is None:
            hessian = jax.vmap(jax.jacfwd(jax.jacrev(net, argnums=0), argnums=0),
                               in_axes=(0, None))(x_all, t)
            negative_laplacians = -jnp.trace(hessian, axis1=-2, axis2=-1)
        else:
            negative_laplacians = D @ fields
        phi, q = fields[:, 0], fields[:, 1]
        velocity = velocity_field(x_all, t, config, xp=jnp).astype(x_all.dtype)
        if flux_divergence is None:
            # Mass shift is constant in the spatial query, but not in time.
            gradient = jax.vmap(jax.grad(lambda x: net(x, t)[0]))(x_all)
            advection = jnp.sum(gradient*velocity, axis=-1)
        else:
            advection = flux_divergence(phi, velocity)
        phase = time_derivative[:, 0] + advection + mobility*negative_laplacians[:, 1]
        chemical = q - phi**3 + phi - epsilon2*negative_laplacians[:, 0]
        return jnp.stack((phase, chemical), axis=-1), phi
    return residuals


def make_loss_fn(x_all, D, config):
    """Full grid, shared-time JVP; optionally change only the residual norm."""
    residuals = make_residual_fn(x_all, D, config)
    weights = jnp.asarray([config.loss.phase_weight, config.loss.chemical_weight,
                           config.loss.mass_weight], dtype=x_all.dtype)
    # Original and extended configuration schemas are deliberately distinct.
    transformed = 'residual' in config and config.residual.mode != 'original'
    schur = (make_schur_transform(D, config, dtype=x_all.dtype)
             if 'preconditioner' in config and config.preconditioner.kind == 'ch_schur' else None)
    causal = (make_causal_transform(D, config, dtype=x_all.dtype)
              if 'preconditioner' in config and config.preconditioner.kind == 'ch_causal' else None)

    def loss_fn(net, times, initial_mean):
        if 'include_initial_time' in config.sample and config.sample.include_initial_time:
            times = jnp.concatenate((jnp.zeros(1, dtype=times.dtype), times))
        residual, phi = jax.vmap(residuals, in_axes=(None, 0))(net, times)
        if schur is not None:
            residual = jax.vmap(schur)(residual)
        if causal is not None:
            residual = causal(residual, times)
        if transformed:
            if config.residual.mode.startswith('map_'):
                jacobian = jax.vmap(lambda t:jax.vmap(net.phase_jacobian,
                    in_axes=(0, None))(x_all, t))(times)
                w = (1+config.phase.leak)/jacobian
            else:
                w = inverse_jacobian_weight(phi, config.residual.eta)
            factors = (jnp.stack((w, jnp.ones_like(w)), axis=-1)
                       if config.residual.mode in ('transport', 'map_transport') else w[..., None])
            residual = residual*factors
        parts = jnp.concatenate((jnp.mean(residual**2, axis=(0, 1)),
                                 jnp.mean((jnp.mean(phi, axis=1)-initial_mean)**2)[None]))
        return jnp.dot(weights, parts), parts
    return loss_fn
