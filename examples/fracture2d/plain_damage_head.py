"""No-reference damage heads with matched initial parameter sensitivities."""
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from scipy.sparse import diags

from examples.fracture2d.compact_damage_features import Readout, residual_column_scale, residual_power_loss
from examples.fracture2d.compact_damage_loading import JointNetwork
from examples.fracture2d.conditioned_damage import _build_network
from examples.fracture2d.plain_sigmoid_damage import make_gradual_problem
from examples.fracture2d.split_pinn_gradients import make_padded_matvec


def initialization(head, damage):
    if not np.isfinite(damage) or not 0 < damage < 1:
        raise ValueError('head comparison requires finite strictly bounded initial damage')
    if head == 'sigmoid':
        latent, tangent = np.log(damage)-np.log1p(-damage), damage*(1-damage)
    elif head == 'exp_square':
        latent = -np.sqrt(-np.log(damage))
        tangent = -2*latent*damage
    elif head == 'identity':
        latent, tangent = damage, 1.
    else:
        raise ValueError(f'unknown explicit damage head: {head}')
    return latent, tangent, damage*(1-damage)/tangent


def map_damage(head, latent):
    if head == 'sigmoid':
        return jax.nn.sigmoid(latent)
    if head == 'exp_square':
        return jnp.exp(-latent**2)
    if head == 'identity':
        return latent
    raise ValueError(f'unknown explicit damage head: {head}')


def fresh_network(c, input_dimension, matrix, *, head):
    latent, _, gain = initialization(head, c.head.initial_damage)
    trunk = _build_network(c, input_dimension)
    trunk = eqx.tree_at(lambda n:n.output.bias, trunk,
        jnp.full_like(trunk.output.bias,latent/gain))
    return JointNetwork(trunk,Readout(jnp.zeros(matrix.shape[1],jnp.float64)))


def initial_scales(matrix, D, H, *, head, initial_damage, gc0, gc1):
    _, tangent, gain = initialization(head, initial_damage)
    scaled = diags(np.full(len(H),tangent))@matrix
    return residual_column_scale(D,scaled,H,gc0=gc0,gc1=gc1),gain


def make_problem(c, s, column_scale, trunk_gain, *, head):
    initialization(head,c.head.initial_damage)
    _,_,_,latent = make_gradual_problem(c,s,column_scale,trunk_gain)
    gc0,gc1 = c.physical.Gc/c.physical.l0,c.physical.Gc*c.physical.l0
    matvec,_ = make_padded_matvec(s['operator'])
    matvec = jax.jit(matvec)

    def image(net,H,load):
        return map_damage(head,latent(net,jax.lax.stop_gradient(H),load))

    def metrics(net,H,load):
        H = jax.lax.stop_gradient(H)
        d = image(net,H,load)
        parts = (gc0*d,gc1*matvec(d),2*H*(d-1))
        return dict(AT2_relative=jnp.max(abs(sum(parts)))/sum(jnp.max(abs(p)) for p in parts),
            damage_min=d.min(),damage_max=d.max())

    def objective(power):
        def loss(net,H,load):
            H = jax.lax.stop_gradient(H)
            d = image(net,H,load)
            r = (gc0+2*H)*d+gc1*matvec(d)-2*H
            return residual_power_loss(r,H,power=power,floor=c.conditioned.loss_floor)
        return loss

    return image,metrics,dict(residual_l32=objective(32),residual_l16=objective(16)),latent
