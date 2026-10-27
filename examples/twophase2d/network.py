"""Periodic mixed-field network with exact prescribed phase initial data only."""
import equinox as eqx
import jax
import jax.numpy as jnp
from nonlocal_pinn.nn.arch import make_pinn


def inverse_leaky_tanh(phi, leak, steps):
    """Inverse of tanh(u)+leak*u on the prescribed IC range[-1,1]."""
    def iterate(_, u):
        value = jnp.tanh(u)
        return u-(value+leak*u-phi)/(1-value**2+leak)
    return jax.lax.fori_loop(0, steps, iterate, phi)


def inverse_cubic(u):
    """Real inverse of u=phi+phi**3; derivative is 1/(1+3*phi**2)."""
    return (2/jnp.sqrt(3.))*jnp.sinh(jnp.arcsinh(1.5*jnp.sqrt(3.)*u)/3)


class PhaseNetwork(eqx.Module):
    net: object
    modes: object
    length: float = eqx.field(static=True)
    duration: float = eqx.field(static=True)
    center: tuple = eqx.field(static=True)
    level: float = eqx.field(static=True)
    width: float = eqx.field(static=True)
    phase_map: str = eqx.field(static=True)
    leak: float = eqx.field(static=True)
    inverse_steps: int = eqx.field(static=True)

    def __init__(self, config, key):
        self.length = config.geometry.length
        self.duration = config.physical.terminal_time
        self.center = tuple(config.initial.center)
        self.level = config.initial.level
        self.width = config.initial.width
        if 'phase' in config:
            self.phase_map = config.phase.map
            if self.phase_map == 'leaky_tanh':
                self.leak = config.phase.leak
                self.inverse_steps = config.phase.inverse_steps
            else:
                self.leak, self.inverse_steps = 0., 0
        else:
            # Original schema; all historical checkpoint leaves are unchanged.
            self.phase_map, self.leak, self.inverse_steps = 'additive', 0., 0
        self.modes = jnp.arange(1, config.arch.spatial_modes+1, dtype=jnp.float32)
        self.net = make_pinn(4*config.arch.spatial_modes, 2,
            hidden_dim=config.arch.hidden_dim, num_layers=config.arch.num_layers,
            activation=config.arch.activation, use_spatial_emb=False,
            use_temporal_emb=True, temporal_emb_dim=config.arch.temporal_emb_dim,
            temporal_scale=config.arch.temporal_scale, key=key)

    def raw(self, x, t):
        # These integer modes impose periodicity; they are not learned Fourier
        # frequencies. Keep the array leaf for direct historical model loading.
        modes = jax.lax.stop_gradient(self.modes)
        angle = 2*jnp.pi*x[:, None]*modes[None]/self.length
        features = jnp.concatenate((jnp.sin(angle).ravel(), jnp.cos(angle).ravel()))
        return self.net(features, t/self.duration)

    def initial(self, x):
        s = jnp.sum(jnp.sin(jnp.pi*(x-jnp.asarray(self.center))/self.length)**2)
        return jnp.tanh((s-self.level)/self.width)

    def phase_jacobian(self, x, t):
        if self.phase_map != 'leaky_tanh':
            raise ValueError('Inverse-map Jacobian requires the explicit leaky_tanh phase map')
        u0 = inverse_leaky_tanh(self.initial(x), self.leak, self.inverse_steps)
        u = u0+(t/self.duration)*self.raw(x, t)[0]
        return 1-jnp.tanh(u)**2+self.leak

    def __call__(self, x, t):
        raw = self.raw(x, t)
        phi0 = self.initial(x)
        correction = (t/self.duration)*raw[0]
        if self.phase_map == 'additive':
            phi = phi0+correction
        elif self.phase_map == 'leaky_tanh':
            u0 = inverse_leaky_tanh(phi0, self.leak, self.inverse_steps)
            phi = phi0+(jnp.tanh(u0+correction)-jnp.tanh(u0))+self.leak*correction
        elif self.phase_map == 'inverse_cubic':
            u0 = phi0+phi0**3
            phi = phi0+(inverse_cubic(u0+correction)-inverse_cubic(u0))
        else:
            raise ValueError(f'Unknown phase map: {self.phase_map}')
        return jnp.stack((phi, raw[1]))
