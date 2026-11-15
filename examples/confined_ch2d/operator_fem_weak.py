"""Full nodal P1/lumped mixed CH residual with chemical-potential elimination."""
import equinox as eqx
import jax
import jax.numpy as jnp

from examples.confined_ch2d.operator_precision import compute_arrays, compute_dtype
from examples.confined_ch2d.operator_fem_assembly import validate_fem
from examples.confined_ch2d.operator_problem import fixed_initial


def action(value, data):
    return jnp.sum(data['fe_coefficients']*(value[:, None]-value[data['fe_neighbors']]), axis=1)


def phase(net, prepared, t, data, c):
    s = t/c.physical.terminal_time
    raw = net.decode(prepared, jnp.reshape(s, (1,)))[0]
    mass = data['fe_mass']
    mean = jnp.dot(mass, raw)/jnp.sum(mass)
    return jnp.where(mass > 0, data['fe_initial']+s*(raw-mean), 0.)


def make_loss(c):
    validate_fem(c)

    def one(net, data, times):
        from examples.confined_ch2d.operator_training import encode
        encoded = encode(net, data)
        prepared = net.prepare_encoded(encoded, data['fe_points'])

        def at(t):
            phi, dt = jax.jvp(lambda tt: phase(net, prepared, tt, data, c), (t,), (jnp.ones_like(t),))
            mu = c.physical.beta*(phi**3-phi)+c.physical.epsilon**2*action(phi, data)
            residual = c.physical.terminal_time*(dt+c.physical.mobility*action(mu, data))
            filtered = data['fe_metric'] @ residual
            return jnp.dot(data['fe_mass'], filtered**2)/jnp.sum(data['fe_mass'])

        times = eqx.error_if(times, times[0] != 0, 'FEM time stream requires its leading initial coordinate')
        fraction = c.loss.initial_residual_fraction
        if fraction == 0.:
            return jax.vmap(at)(times[1:]).mean()
        values = jax.vmap(at)(times)
        return fraction*values[0]+(1-fraction)*values[1:].mean()

    def loss(net, batch, times):
        batch, times = compute_arrays(batch, c), jnp.asarray(times, dtype=compute_dtype(c))
        return jax.vmap(one, in_axes=(None, 0, None))(net, batch, times).mean()
    return loss


def fields(net, data, times, c):
    from examples.confined_ch2d.operator_training import encode
    data, times = compute_arrays(data, c), jnp.asarray(times, dtype=compute_dtype(c))
    encoded = encode(net, data)
    grid = net.prepare_encoded(encoded, data['fe_points'])
    query = net.prepare_encoded(encoded, data['points'])

    def at(t):
        s = t/c.physical.terminal_time
        value = net.decode(grid, jnp.reshape(s, (1,)))[0]
        mean = jnp.dot(data['fe_mass'], value)/jnp.sum(data['fe_mass'])
        nodal_phi = jnp.where(data['fe_mass'] > 0, data['fe_initial']+s*(value-mean), 0.)
        nodal_mu = c.physical.beta*(nodal_phi**3-nodal_phi)+c.physical.epsilon**2*action(nodal_phi, data)
        # All common evaluation nodes are centroids of these same triangles.
        phi = data['initial']+s*(net.decode(query, jnp.reshape(s, (1,)))[0]-mean)
        mu = nodal_mu[data['fe_cells']].mean(axis=1)
        return jnp.where(data['volumes'][:, None] > 0, jnp.stack((phi, mu), axis=-1), 0.)
    return jax.vmap(at)(times)


def query_phase_encoded(net, encoded, data, points, times, c):
    """Off-grid neural phase keeps the FEM nodal global mass quadrature."""
    grid = net.prepare_encoded(encoded, data['fe_points'])
    query = net.prepare_encoded(encoded, points)
    s = times/c.physical.terminal_time
    raw_grid = net.decode(grid, s)
    means = raw_grid @ data['fe_mass']/jnp.sum(data['fe_mass'])
    return fixed_initial(points,c,xp=jnp)[None,:]+s[:,None]*(net.decode(query,s)-means[:,None])
