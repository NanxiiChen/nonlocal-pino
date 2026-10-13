"""Persistent Adam updates for exact core+halo damage objectives."""
import equinox as eqx
import jax
import numpy as np
import optax


def make_adam(objective, *, learning_rate, decay_steps, decay_rate, minimum_rate,
              b1, b2, eps, eps_root, grad_clip):
    values=(learning_rate,decay_rate,minimum_rate,b1,b2,eps,eps_root,grad_clip)
    if (not np.isfinite(values).all() or learning_rate <= 0
            or type(decay_steps) is not int or decay_steps < 1
            or not 0 < decay_rate <= 1 or not 0 < minimum_rate <= learning_rate
            or not 0 < b1 < 1 or not 0 < b2 < 1 or eps <= 0
            or eps_root < 0 or grad_clip <= 0):
        raise ValueError('Adam requires explicit finite rate/schedule/moments/epsilon/clip controls')
    schedule=optax.exponential_decay(init_value=learning_rate,transition_steps=decay_steps,
        decay_rate=decay_rate,end_value=minimum_rate)
    optimizer=optax.chain(optax.clip_by_global_norm(grad_clip),
        optax.adam(schedule,b1=b1,b2=b2,eps=eps,eps_root=eps_root))

    @eqx.filter_jit
    def step(net,state,context,load):
        value,gradient=eqx.filter_value_and_grad(objective)(net,context,load)
        gradient_norm=optax.global_norm(gradient)
        updates,state=optimizer.update(gradient,state,eqx.filter(net,eqx.is_inexact_array))
        return eqx.apply_updates(net,updates),state,value,gradient_norm

    return optimizer,step,schedule
