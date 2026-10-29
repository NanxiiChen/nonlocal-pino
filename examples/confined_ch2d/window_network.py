"""Neural increments on fixed quadrature; previous states are frozen PDE outputs."""
import equinox as eqx
import jax,jax.numpy as jnp
from nonlocal_pinn.nn.arch import make_pinn


class WindowNetwork(eqx.Module):
    net:object
    center:tuple=eqx.field(static=True)
    axes:tuple=eqx.field(static=True)
    modes:tuple=eqx.field(static=True)

    def __init__(self,c,key):
        self.center=tuple(c.geometry.center);self.axes=tuple(c.geometry.axes)
        self.modes=tuple(range(1,c.arch.spatial_modes+1))
        self.net=make_pinn(2+4*len(self.modes),1,hidden_dim=c.arch.hidden_dim,
            num_layers=c.arch.num_layers,activation=c.arch.activation,use_spatial_emb=False,
            use_temporal_emb=True,temporal_emb_dim=c.arch.temporal_emb_dim,
            temporal_scale=c.arch.temporal_scale,key=key)

    def __call__(self,x,s):
        z=(x-jnp.asarray(self.center))/jnp.asarray(self.axes)
        angle=jnp.pi*z[:,None]*jnp.asarray(self.modes)
        return self.net(jnp.concatenate((z,jnp.sin(angle).ravel(),jnp.cos(angle).ravel())),s)[0]


def build_window_network(c,key):
    with jax.enable_x64(False):net=WindowNetwork(c,key)
    return jax.tree.map(lambda a:a.astype(jnp.float64) if eqx.is_inexact_array(a) else a,net)


def window_fields(net,points,t,width,initial,mean,volumes,D,c):
    s=t/width
    change=jax.vmap(net,in_axes=(0,None))(points,s)
    phi=initial+s*change
    phi=phi+(mean-jnp.dot(volumes,phi)/volumes.sum())
    mu=c.physical.beta*(phi**3-phi)+c.physical.epsilon**2*(D@phi)
    return jnp.stack((phi,mu),axis=-1)


def make_window_loss(points,volumes,D,mean,c):
    def loss(net,initial,width,times,metric):
        if c.sample.include_initial_time:times=jnp.concatenate((jnp.zeros(1,dtype=times.dtype),times))
        def at(t):
            field,dt=jax.jvp(lambda tt:window_fields(net,points,tt,width,initial,mean,volumes,D,c),(t,),(jnp.ones_like(t),))
            residual=width*(dt[:,0]+c.physical.mobility*(D@field[:,1]))
            transformed=metric(residual)
            return jnp.dot(volumes,transformed**2)/volumes.sum()
        return c.loss.phase_weight*jnp.mean(jax.vmap(at)(times))
    return loss
