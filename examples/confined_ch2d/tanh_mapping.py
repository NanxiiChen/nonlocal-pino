"""Mass-preserving tanh map and its implicitly differentiated scalar offset."""
from functools import partial
import equinox as eqx
import jax
import jax.numpy as jnp


def tanh_jacobian(z,amplitude):
    """a*sech²(z), avoiding subtraction of two nearly equal pure-phase values."""
    e=jnp.exp(-2*jnp.abs(z))
    return amplitude*4*e/(1+e)**2


@jax.custom_jvp
def stable_tanh(z):
    """Rounded tanh value with its cancellation-free mathematical derivative."""
    return jnp.tanh(z)


@stable_tanh.defjvp
def _stable_tanh_jvp(primals,tangents):
    (z,),(dz,)=primals,tangents
    return stable_tanh(z),tanh_jacobian(z,1.)*dz


@partial(jax.custom_jvp,nondiff_argnums=(3,))
def mass_shift(latent,volumes,mean,amplitude):
    """Unique offset with weighted mean a*tanh(latent+offset)=mean."""
    if latent.ndim!=1 or volumes.shape!=latent.shape:raise ValueError('Tanh mass map needs matching space vectors')
    latent=eqx.error_if(latent,jnp.any(~jnp.isfinite(latent)),'Nonfinite tanh latent field')
    mean=eqx.error_if(mean,jnp.abs(mean)>=amplitude,'Tanh mass target must lie strictly inside its amplitude')
    center=jnp.arctanh(mean/amplitude)
    low=center-jnp.max(latent);high=center-jnp.min(latent)
    total=jnp.sum(volumes)
    def bisect(_,bounds):
        lo,hi=bounds;mid=(lo+hi)/2
        below=jnp.dot(volumes,amplitude*jnp.tanh(latent+mid))/total<mean
        return jnp.where(below,mid,lo),jnp.where(below,hi,mid)
    lo,hi=jax.lax.fori_loop(0,64,bisect,(low,high))
    return (lo+hi)/2


@mass_shift.defjvp
def _mass_shift_jvp(amplitude,primals,tangents):
    latent,volumes,mean=primals;dlatent,dvolumes,dmean=tangents
    offset=mass_shift(latent,volumes,mean,amplitude)
    phi=amplitude*stable_tanh(latent+offset);jac=tanh_jacobian(latent+offset,amplitude)
    denominator=jnp.dot(volumes,jac)
    derivative=(jnp.sum(volumes)*dmean-jnp.dot(volumes*jac,dlatent)-jnp.dot(dvolumes,phi-mean))/denominator
    return offset,derivative


def phase_state(net,points,t,mean,volumes):
    """Physical projected phase, diagonal tanh Jacobian, and scalar offset."""
    latent=jax.vmap(net.latent,in_axes=(0,None))(points,t)
    offset=mass_shift(latent,volumes,mean,net.tanh_amplitude)
    z=latent+offset
    return net.tanh_amplitude*stable_tanh(z),tanh_jacobian(z,net.tanh_amplitude),offset
