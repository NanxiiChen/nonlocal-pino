"""Time microbatches share parameters and precede a single optimizer update."""
import equinox as eqx
import jax
import jax.numpy as jnp

from examples.channel_nsch2d.losses import part_names


def make_accumulated_fn(c, loss_fn):
    """Return ((mean loss, mean parts), mean gradient) for the whole time batch.

    Equal-sized blocks are only a memory bound. Neither clipping nor optimizer
    state updates occur here; every block differentiates the same model.
    """
    size=c.train.microbatch.size
    blocks=c.sample.Nt//size
    value_grad=eqx.filter_value_and_grad(loss_fn,has_aux=True)

    def accumulated(model,batch,times):
        if times.shape!=(c.sample.Nt,):
            raise ValueError('Time accumulation requires the complete configured Nt batch')
        zero=jax.tree.map(jnp.zeros_like,eqx.filter(model,eqx.is_inexact_array))
        initial=(jnp.zeros((),dtype=times.dtype),
                 jnp.zeros(len(part_names(c)),dtype=times.dtype),zero)

        def add(carry,ts):
            total,parts,gradient=carry
            (value,vector),grad=value_grad(model,batch,ts)
            gradient=jax.tree.map(lambda a,b:a+b,gradient,grad)
            return (total+value,parts+vector,gradient),None

        (total,parts,gradient),_=jax.lax.scan(add,initial,times.reshape(blocks,size))
        gradient=jax.tree.map(lambda a:a/blocks,gradient)
        return (total/blocks,parts/blocks),gradient

    return accumulated
