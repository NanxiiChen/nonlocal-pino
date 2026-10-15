"""Direct sigmoid unknown with no frozen prediction or reference output field."""
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from examples.fracture2d._legacy.compact_damage_features import pack_basis, residual_power_loss
from examples.fracture2d._legacy.split_pinn_gradients import make_padded_matvec


def make_problem(c, s, column_scale, trunk_gain):
    """Preserved single-load/transfer decoder with load-squared latent scaling."""
    return _make_problem(c, s, column_scale, trunk_gain, scale_logits_with_load=True)


def make_gradual_problem(c, s, column_scale, trunk_gain):
    """Plain trainable logits: changing load does not rescale the learned bias."""
    return _make_problem(c, s, column_scale, trunk_gain, scale_logits_with_load=False)


def _make_problem(c, s, column_scale, trunk_gain, *, scale_logits_with_load):
    gc0, gc1 = c.physical.Gc/c.physical.l0, c.physical.Gc*c.physical.l0
    if (not np.isfinite(trunk_gain) or trunk_gain <= 0
            or column_scale.shape != (s['matrix'].shape[1],)
            or not np.isfinite(column_scale).all() or np.min(column_scale) <= 0):
        raise ValueError('plain sigmoid requires explicit finite positive parameter scales')
    phi, _ = pack_basis(s['matrix'], column_scale)
    matvec, _ = make_padded_matvec(s['operator'])
    matvec = jax.jit(matvec)

    def latent(net, H, load):
        H = jax.lax.stop_gradient(H)
        trunk_factor = trunk_gain*c.physical.E/gc0*load**2 if scale_logits_with_load else trunk_gain
        readout_factor = (load/s['reference_load'])**2 if scale_logits_with_load else 1.
        return (trunk_factor
                *jax.vmap(net.trunk)(s['features'](H, load))[:, 0]
                +readout_factor*phi(net.readout))

    def image(net, H, load):
        return jax.nn.sigmoid(latent(net, H, load))

    def residual(net, H, load):
        H = jax.lax.stop_gradient(H)
        d = image(net, H, load)
        return (gc0+2*H)*d+gc1*matvec(d)-2*H

    def metrics(net, H, load):
        H = jax.lax.stop_gradient(H)
        d = image(net, H, load)
        parts = (gc0*d, gc1*matvec(d), 2*H*(d-1))
        return dict(AT2_relative=jnp.max(abs(sum(parts)))/sum(jnp.max(abs(p)) for p in parts),
                    damage_min=jnp.min(d), damage_max=jnp.max(d))

    def peak(power):
        def objective(net, H, load):
            H = jax.lax.stop_gradient(H)
            return residual_power_loss(residual(net, H, load), H, power=power,
                                       floor=c.conditioned.loss_floor)
        return objective

    return image, metrics, dict(residual_l32=peak(32), residual_l16=peak(16)), latent


def recenter_bias(net, latent, H, load, teacher_damage, *, trunk_factor):
    """One-time learned bias initialization; all existing weights stay trainable."""
    target = np.asarray(teacher_damage)
    if (target.shape != H.shape or not np.isfinite(target).all()
            or np.min(target) <= 0 or np.max(target) >= 1
            or not np.isfinite(trunk_factor) or trunk_factor <= 0):
        raise ValueError('plain-head transfer requires finite strictly bounded own-neural image')
    target_logit = jnp.log(teacher_damage)-jnp.log1p(-teacher_damage)
    shift = jnp.mean(target_logit-latent(net, H, load))/trunk_factor
    candidate = eqx.tree_at(lambda n: n.trunk.output.bias, net, net.trunk.output.bias+shift)
    return candidate, target_logit, float(shift)


def latent_basis_scaling(net, matrix, old_scale):
    """Normalize latent columns and exactly preserve their represented field."""
    rms = np.sqrt(np.asarray(matrix.power(2).sum(axis=0)).ravel()/matrix.shape[0])
    if (old_scale.shape != rms.shape or not np.isfinite(rms).all() or np.min(rms) <= 0
            or not np.isfinite(old_scale).all() or np.min(old_scale) <= 0):
        raise ValueError('latent-basis normalization requires finite nonzero columns/scales')
    new_scale = 1/rms
    coefficients = net.readout.coefficients*jnp.asarray(old_scale/new_scale)
    converted = eqx.tree_at(lambda n:n.readout.coefficients, net, coefficients)
    return converted, new_scale


def make_transfer_objective(latent, target_logit):
    """Own-network parameter conversion; readiness is still the AT2 residual."""
    target_logit = jax.lax.stop_gradient(target_logit)
    if (not np.isfinite(np.asarray(target_logit)).all()
            or float(jnp.mean(target_logit**2)) <= 0):
        raise ValueError('own-neural transfer logits must be finite with nonzero mean square')

    def objective(net, H, load):
        difference = latent(net, H, load)-target_logit
        return jnp.mean(difference**2)/jnp.mean(target_logit**2)

    return objective
