"""Optional two-loss training; adaptive weights are optimizer state, not AD inputs.

Gradient-norm balancing follows the existing corrosion implementation: inverse
L1 norms over all trainable parameters. It is not relative-rate GradNorm.
Staggering alternates losses, without freezing a field or shared parameters.
"""
import equinox as eqx
import jax
import jax.numpy as jnp


def active_losses(iteration,settings):
    if settings.kind!='stagger':return jnp.ones(2,dtype=jnp.bool_)
    stage=(iteration//settings.switch_every)%2
    return jnp.where(iteration<settings.stagger_steps,
        jnp.arange(2)==stage,jnp.ones(2,dtype=jnp.bool_))


def make_balanced_step(loss,optimizer,c):
    settings=c.train.loss_balance
    bias=jnp.asarray([c.loss.phase_weight,c.loss.chemical_weight])

    @eqx.filter_jit
    def step(net,state,weights,norms,times,mean,iteration):
        parts,pullback=eqx.filter_vjp(lambda model:loss(model,times,mean)[1],net)
        if settings.kind=='gradient_norm':
            def refresh(_):
                values=[]
                for i in range(2):
                    grad=pullback(jax.nn.one_hot(i,2,dtype=parts.dtype))[0]
                    values.append(sum(jnp.sum(jnp.abs(a)) for a in jax.tree.leaves(grad) if eqx.is_array(a)))
                raw=jnp.stack(values)
                raw=eqx.error_if(raw,~jnp.all(jnp.isfinite(raw)),'Nonfinite balancing gradient norm')
                measured=jnp.clip(raw,settings.norm_floor,settings.norm_ceiling)
                proposed=jnp.clip(jnp.mean(measured)/measured,settings.weight_floor,settings.weight_cap)
                updated=settings.alpha*proposed+(1-settings.alpha)*weights
                return jax.lax.stop_gradient(updated),jax.lax.stop_gradient(measured)
            weights,norms=jax.lax.cond(iteration%settings.update_every==0,refresh,
                lambda _:(weights,norms),operand=None)
        active=active_losses(iteration,settings)
        effective=jax.lax.stop_gradient(weights*bias*active)
        grad=pullback(effective)[0]
        finite=jnp.all(jnp.isfinite(parts)) & jnp.all(jnp.isfinite(weights)) & jnp.all(weights>0)
        for a in jax.tree.leaves(grad):
            if eqx.is_array(a):finite=finite & jnp.all(jnp.isfinite(a))
        total=jnp.where(finite,jnp.dot(parts,effective),jnp.nan)
        updates,state=optimizer.update(grad,state,net)
        return eqx.apply_updates(net,updates),state,weights,norms,total,parts,effective,active
    return step


def frozen_joint_loss(loss,weights,c):
    """Keep a fixed scalar and its true gradient throughout line searches."""
    effective=jax.lax.stop_gradient(weights*jnp.asarray([c.loss.phase_weight,c.loss.chemical_weight]))
    def objective(net,times,mean):
        _,parts=loss(net,times,mean)
        return jnp.dot(effective,parts),parts
    return objective
