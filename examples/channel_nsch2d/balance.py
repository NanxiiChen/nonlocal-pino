"""Stopped inverse-gradient-norm balancing, following this repo's KKS convention.

This balances L1 norms with a clipped inverse ratio and EMA. It is not the
learned-weight/training-rate objective of the original GradNorm paper.
"""
import equinox as eqx
import jax
import jax.numpy as jnp
from examples.channel_nsch2d.losses import make_loss_fn,part_names


def group_names(c):
    if 'balance' in c and 'grouping' in c.balance and c.balance.grouping=='terms':
        return part_names(c)
    names=('phase','chemical','momentum')
    if 'velocity' not in c:names+=('continuity',)
    return names+('boundary',)


def group_losses(parts,c):
    if 'balance' in c and 'grouping' in c.balance and c.balance.grouping=='terms':
        return parts
    values=[parts[0],parts[1],parts[2]+parts[3]]
    start=4
    if 'velocity' not in c:values.append(parts[4]);start=5
    values.append(jnp.sum(parts[start:]))
    return jnp.stack(values)


def weights_from_norms(norms,previous,settings):
    """Legacy clipped inverse-L1 ratio/EMA, stopped before model differentiation."""
    safe=jnp.clip(norms,settings.epsilon,1/settings.epsilon)
    target=jnp.clip(jnp.mean(safe)/safe,settings.epsilon,settings.cap)
    return jax.lax.stop_gradient(
        settings.new_weight_fraction*target+(1-settings.new_weight_fraction)*previous)


def balanced_value_and_grad(fn,model,previous,refresh,settings):
    """Reuse one VJP tape; sequential norm probes bound memory at refreshes."""
    values,pullback,aux=eqx.filter_vjp(fn,model,has_aux=True)
    if values.shape!=previous.shape:raise ValueError('One adaptive weight per explicit group required')
    def update(_):
        def norm(cotangent):
            gradient=pullback(cotangent)[0]
            return sum(jnp.sum(jnp.abs(a)) for a in jax.tree.leaves(gradient) if eqx.is_array(a))
        norms=jax.lax.map(norm,jnp.eye(len(values),dtype=values.dtype))
        return weights_from_norms(norms,previous,settings),norms
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


def make_balanced_accumulated_fn(c):
    """Balance the FULL mean gradient of each term, then take one joint update.

    Refresh: accumulate each term's parameter gradient across every time block,
    then take its L1 norm. Reuse these gradients for the newly weighted update.
    Other steps: accumulate only the weighted gradient with fixed weights.
    Taking norms before block averaging would implement a different algorithm.
    """
    if ('grouping' not in c.balance or c.balance.grouping!='terms'
            or 'microbatch' not in c.train):
        raise ValueError('Balanced time accumulation requires explicit term grouping and microbatch')
    loss=make_loss_fn(c);size=c.train.microbatch.size;blocks=c.sample.Nt//size
    count=len(group_names(c))

    def terms(model,batch,times):
        _,parts=loss(model,batch,times)
        return group_losses(parts,c),parts

    def weighted(model,batch,times,weights):
        values,parts=terms(model,batch,times)
        return jnp.dot(weights,values),parts
    weighted_grad=eqx.filter_value_and_grad(weighted,has_aux=True)

    def apply(model,batch,times,previous,refresh):
        if times.shape!=(c.sample.Nt,) or previous.shape!=(count,):
            raise ValueError('Balanced accumulation requires complete times and one weight per term')
        previous=jax.lax.stop_gradient(previous)
        zero=jax.tree.map(jnp.zeros_like,eqx.filter(model,eqx.is_inexact_array))
        empty=jnp.zeros(count,dtype=times.dtype)

        def update(_):
            jaczero=jax.tree.map(lambda a:jnp.zeros((count,*a.shape),dtype=a.dtype),zero)

            def add(carry,ts):
                values,parts,jacobian=carry
                vector,pullback,aux=eqx.filter_vjp(lambda n:terms(n,batch,ts),model,has_aux=True)
                jac=jax.lax.map(lambda cotangent:pullback(cotangent)[0],
                    jnp.eye(count,dtype=times.dtype))
                return (values+vector,parts+aux,jax.tree.map(lambda a,b:a+b,jacobian,jac)),None

            (values,parts,jacobian),_=jax.lax.scan(add,(empty,empty,jaczero),times.reshape(blocks,size))
            values,parts=values/blocks,parts/blocks
            jacobian=jax.tree.map(lambda a:a/blocks,jacobian)
            norms=sum(jnp.sum(jnp.abs(a).reshape(count,-1),axis=1)
                      for a in jax.tree.leaves(jacobian) if eqx.is_array(a))
            weights=weights_from_norms(norms,previous,c.balance)
            gradient=jax.tree.map(lambda a:jnp.tensordot(weights,a,axes=1),jacobian)
            return jnp.dot(weights,values),parts,weights,norms,gradient

        def held(_):
            def add(carry,ts):
                total,parts,gradient=carry
                (value,vector),grad=weighted_grad(model,batch,ts,previous)
                return (total+value,parts+vector,jax.tree.map(lambda a,b:a+b,gradient,grad)),None
            (total,parts,gradient),_=jax.lax.scan(add,
                (jnp.zeros((),dtype=times.dtype),empty,zero),times.reshape(blocks,size))
            return total/blocks,parts/blocks,previous,empty,jax.tree.map(lambda a:a/blocks,gradient)

        return jax.lax.cond(refresh,update,held,None)
    return apply
