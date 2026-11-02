"""Stopped inverse-gradient-norm balancing, following this repo's KKS convention.

This balances L1 norms with a clipped inverse ratio and EMA. It is not the
learned-weight/training-rate objective of the original GradNorm paper.
"""
import equinox as eqx
import jax
import jax.numpy as jnp
from examples.channel_nsch2d.losses import make_loss_fn


def group_names(c):
    names=('phase','chemical','momentum')
    if 'velocity' not in c:names+=('continuity',)
    return names+('boundary',)


def group_losses(parts,c):
    values=[parts[0],parts[1],parts[2]+parts[3]]
    start=4
    if 'velocity' not in c:values.append(parts[4]);start=5
    values.append(jnp.sum(parts[start:]))
    return jnp.stack(values)


def balanced_value_and_grad(fn,model,previous,refresh,settings):
    """Reuse one VJP tape; sequential norm probes bound memory at refreshes."""
    values,pullback,aux=eqx.filter_vjp(fn,model,has_aux=True)
    if values.shape!=previous.shape:raise ValueError('One adaptive weight per explicit group required')
    def update(_):
        def norm(cotangent):
            gradient=pullback(cotangent)[0]
            return sum(jnp.sum(jnp.abs(a)) for a in jax.tree.leaves(gradient) if eqx.is_array(a))
        norms=jax.lax.map(norm,jnp.eye(len(values),dtype=values.dtype))
        safe=jnp.clip(norms,settings.epsilon,1/settings.epsilon)
        target=jnp.clip(jnp.mean(safe)/safe,settings.epsilon,settings.cap)
        weights=settings.new_weight_fraction*target+(1-settings.new_weight_fraction)*previous
        return weights,norms
    weights,norms=jax.lax.cond(refresh,update,lambda _:(previous,jnp.zeros_like(previous)),None)
    weights=jax.lax.stop_gradient(weights)
    gradient=pullback(weights)[0]
    return jnp.dot(weights,values),aux,weights,norms,gradient


def make_balanced_fn(c):
    loss=make_loss_fn(c)
    def apply(model,batch,times,previous,refresh):
        def terms(net):
            _,parts=loss(net,batch,times)
            return group_losses(parts,c),parts
        return balanced_value_and_grad(terms,model,previous,refresh,c.balance)
    return apply
