"""Periodic phase/chemical PINN, with IC and discrete mass constraints."""
import equinox as eqx
import jax
import jax.numpy as jnp
from nonlocal_pinn.nn.arch import make_pinn
from examples.precipitate2d.problem import multiple_initial_field


class PhaseNetwork(eqx.Module):
    net: object
    modes: tuple = eqx.field(static=True)
    length: float = eqx.field(static=True)
    duration: float = eqx.field(static=True)
    initial_kind: str = eqx.field(static=True)
    centers: tuple = eqx.field(static=True)
    levels: tuple = eqx.field(static=True)
    width: float = eqx.field(static=True)
    background: float = eqx.field(static=True)

    def __init__(self, c, key):
        self.modes = tuple(range(1,c.arch.spatial_modes+1))
        self.length, self.duration = c.geometry.length, c.physical.terminal_time
        if 'kind' in c.initial:
            self.initial_kind = c.initial.kind
            self.centers = tuple(tuple(p) for p in c.initial.centers)
            self.levels = tuple(c.initial.levels)
        else:
            self.initial_kind = 'single'
            self.centers, self.levels = (tuple(c.initial.center),), (c.initial.level,)
        self.width, self.background = c.initial.width, c.initial.background
        self.net = make_pinn(4*len(self.modes), 2, hidden_dim=c.arch.hidden_dim,
            num_layers=c.arch.num_layers, activation=c.arch.activation, use_spatial_emb=False,
            use_temporal_emb=True, temporal_emb_dim=c.arch.temporal_emb_dim,
            temporal_scale=c.arch.temporal_scale, key=key)

    def __call__(self, x, t):
        angle = 2*jnp.pi*x[:,None]*jnp.asarray(self.modes)[None]/self.length
        raw = self.net(jnp.concatenate((jnp.sin(angle).ravel(),jnp.cos(angle).ravel())),t/self.duration)
        if self.initial_kind == 'multiple':
            p0 = multiple_initial_field(x, self.centers, self.levels, self.width,
                                        self.background, self.length, xp=jnp)
        else:
            s = jnp.sum(jnp.sin(jnp.pi*(x-jnp.asarray(self.centers[0]))/self.length)**2)
            p0 = self.background+(1-self.background)*.5*(1-jnp.tanh((s-self.levels[0])/self.width))
        a = p0+p0**3+(t/self.duration)*raw[0]
        inverse = lambda v:(2/jnp.sqrt(3.))*jnp.sinh(jnp.arcsinh(1.5*jnp.sqrt(3.)*v)/3)
        phi = p0+inverse(a)-inverse(p0+p0**3)
        return jnp.stack((phi, raw[1]))


def predict_fields(net, points, t, initial_mean):
    fields = jax.vmap(net,in_axes=(0,None))(points,t)
    # All physical evaluations use the same full-grid mass projection.
    return fields.at[:,0].add(initial_mean-jnp.mean(fields[:,0]))
