"""Positive, stopped phase-region residual metric; no change to physical phi."""
import jax
import jax.numpy as jnp


def region_weights(phi,measure,settings):
    """Retain a full-domain floor and equalize three soft phase-region budgets.

    For <f>=sum(V*f)/sum(V), region a_k>=0 gives a_k/max(<a_k>,floor).
    W=(1-fraction)+fraction/3*sum_k(a_k/max(<a_k>,floor)) is positive.
    Thus the zero set of the PDE loss is unchanged. Only this metric clips phi;
    the output, physical residual and evaluation fields are never clipped.
    """
    if phi.ndim!=2 or measure.shape!=phi.shape[1:]:
        raise ValueError('Phase metric requires aligned(time,points) and point measures')
    phase=jax.lax.stop_gradient(jnp.clip(phi,-1.,1.))
    regions=jnp.stack(((1-phase)/2,(1+phase)/2,1-phase**2),axis=-1)
    fractions=jnp.sum(regions*measure[None,:,None],axis=1)/jnp.sum(measure)
    normalized=regions/jnp.maximum(fractions[:,None,:],settings.minimum_fraction)
    weight=(1-settings.region_fraction)+settings.region_fraction*jnp.mean(normalized,axis=-1)
    return jax.lax.stop_gradient(weight)
