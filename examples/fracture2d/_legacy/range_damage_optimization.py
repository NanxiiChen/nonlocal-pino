"""Linear-head range optimization; the physical AT2 residual stays unchanged."""
import jax
import jax.numpy as jnp


def damage_diagnostics(d,H,matvec,*,gc0,gc1,volumes,thresholds):
    """Positive-history diagnostics, separating activity from equation balance."""
    H = jax.lax.stop_gradient(H)
    parts = (gc0*d,gc1*matvec(d),2*H*(d-1))
    raw = jnp.max(abs(sum(parts)))
    denominator = sum(jnp.max(abs(part)) for part in parts)
    forcing = jnp.max(2*H)
    values = dict(AT2_relative=raw/denominator,AT2_absolute_inf=raw,
        AT2_part_denominator=denominator,AT2_forcing_inf=forcing,
        AT2_forcing_relative=raw/forcing,damage_min=d.min(),damage_max=d.max(),
        damage_volume_mean=jnp.sum(volumes*d)/jnp.sum(volumes),
        damage_volume_L2=jnp.sqrt(jnp.sum(volumes*d**2)/jnp.sum(volumes)),
        damage_negative_fraction=jnp.mean(d<0),
        AT2_local_part_inf=jnp.max(abs(parts[0])),
        AT2_diffusion_part_inf=jnp.max(abs(parts[1])),
        AT2_history_part_inf=jnp.max(abs(parts[2])))
    for threshold in thresholds:
        values[f'damage_fraction_ge_{threshold:g}'] = jnp.mean(d>=threshold)
    return values


def range_constraints(d,H,*,gc0):
    """Bounds in the same amplitude units as the normalized AT2 forcing."""
    H = jax.lax.stop_gradient(H)
    scale = 2*jnp.sqrt(jnp.mean(H**2))/gc0
    return -d/scale,(d-1)/scale


def augmented_term(d,H,lower,upper,penalty,*,gc0):
    H,lower,upper,penalty = jax.tree.map(jax.lax.stop_gradient,(H,lower,upper,penalty))
    low,high = range_constraints(d,H,gc0=gc0)
    terms = ((jnp.maximum(lower+penalty*low,0)**2-lower**2)
             +(jnp.maximum(upper+penalty*high,0)**2-upper**2))
    return jnp.mean(terms)/(2*penalty)


def update_duals(d,H,lower,upper,penalty,*,gc0):
    d,H,lower,upper,penalty = jax.tree.map(jax.lax.stop_gradient,(d,H,lower,upper,penalty))
    low,high = range_constraints(d,H,gc0=gc0)
    return jnp.maximum(lower+penalty*low,0),jnp.maximum(upper+penalty*high,0)


def make_augmented_objective(image,residual_objective,*,gc0):
    def objective(net,context,load):
        H,lower,upper,penalty = context
        H = jax.lax.stop_gradient(H)
        return (residual_objective(net,H,load)
            +augmented_term(image(net,H,load),H,lower,upper,penalty,gc0=gc0))
    return objective
