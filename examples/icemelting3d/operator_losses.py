"""Original physical AC equation with hard IC and optional exact tanh metric."""
import jax
import jax.numpy as jnp

from examples.icemelting3d.losses import _tanh_difference_ratio
from examples.icemelting3d.operator_data import physics


def latent_fields(net, encoded, points, initial_latent, times):
    prepared = net.prepare_encoded(encoded,points,jnp.zeros((len(points),0),dtype=points.dtype))
    def at(ts):
        return initial_latent[None,:]+ts[:,None]*net.decode(prepared,ts)[...,0]
    return jax.jvp(at,(times,),(jnp.ones_like(times),))


def make_loss_fn(c):
    n1,n2,n3,_ = physics(c)
    def loss(net,times,images,initial,points,sub,mask):
        row,col,values,_ = sub
        def per_condition(image,u0):
            encoded = net.encode(image)
            u,ut = latent_fields(net,encoded,points,u0,times)
            phi = jnp.tanh(u)
            if c.phase.formulation == 'ic_latent':
                ratio = _tanh_difference_ratio(u[:,row],u[:,col])
                diffusion = jax.ops.segment_sum((-values[None,:]*ratio).T,row,num_segments=len(points)).T
                residual = ut+n1*diffusion-n2*phi+n3
            else:
                diffusion = jax.ops.segment_sum((-values[None,:]*(phi[:,row]-phi[:,col])).T,row,num_segments=len(points)).T
                residual = (1-phi*phi)*ut+n1*diffusion+n2*(phi**3-phi)+n3*(1-phi*phi)
            return jnp.sum(jnp.abs(residual)*mask[None,:])/(jnp.sum(mask)*len(times))
        return jax.vmap(per_condition)(images,initial).mean()
    return loss
