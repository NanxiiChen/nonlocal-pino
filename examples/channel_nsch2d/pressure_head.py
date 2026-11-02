"""Pressure-linear feature basis for a frozen physical-field block diagnostic."""
import jax
import jax.numpy as jnp
import numpy as np
from examples.channel_nsch2d.network import temporal_features


def features(net,encoded,points,times,c):
    """Basis of the learned pressure coordinate, including its bias.

    With fixed phi/mu this also gives the physical-pressure response to a row
    change for the optional Pi=p+phi*mu coordinate.
    """
    if 'velocity' not in c or net.output.weight.shape[0]!=4:
        raise ValueError('Pressure-head diagnostic requires the four-output streamfunction model')
    h=net.prepare_encoded(encoded,points)
    modulation=jax.vmap(net.time)(temporal_features(net,times,c)).reshape(
        len(times),len(net.layers),2,net.width)
    hidden=jnp.broadcast_to(h,(len(times),*h.shape))
    for i,layer in enumerate(net.layers):
        hidden=jax.nn.gelu(jax.vmap(jax.vmap(layer))(hidden)*(1+modulation[:,i,0,None,:])
                            +modulation[:,i,1,None,:])
    basis=jnp.concatenate((hidden,jnp.ones((*hidden.shape[:-1],1),dtype=hidden.dtype)),axis=-1)
    return c.loss.pressure_scale*net.output_scale*basis


def feature_jets(net,encoded,points,times,c):
    fn=lambda x:features(net,encoded,x,times,c)
    ex=jnp.broadcast_to(jnp.array([1.,0.],dtype=points.dtype),points.shape)
    ey=jnp.broadcast_to(jnp.array([0.,1.],dtype=points.dtype),points.shape)
    value,dx=jax.jvp(fn,(points,),(ex,));_,dy=jax.jvp(fn,(points,),(ey,))
    return value,dx,dy


def quadratic_statistics(basis,residual,measure):
    """Host FP64 quadratic for sum_i measure_i*(residual_i+basis_i@delta)^2."""
    basis=np.asarray(basis,dtype=np.float64)
    residual=np.asarray(residual,dtype=np.float64)
    measure=np.asarray(measure,dtype=np.float64)
    if basis.ndim!=2 or residual.shape!=(len(basis),) or measure.shape!=residual.shape:
        raise ValueError('Quadratic rows, residual and measure must be aligned')
    if (not all(np.isfinite(a).all() for a in (basis,residual,measure))
            or np.any(measure<0) or np.sum(measure)<=0):
        raise ValueError('Finite rows and nonnegative, nonzero physical measure required')
    weighted=basis*measure[:,None]
    return basis.T@weighted,basis.T@(measure*residual),float(np.sum(measure*residual**2))
