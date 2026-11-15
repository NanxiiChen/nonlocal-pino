"""Conservative CH coordinate with exact, compact two-hop operator support.

For positive D approximating -Laplacian and alpha=epsilon^2/beta:
  phi = phi0 + alpha D chi, chi(0)=0,
  mu = beta(phi^3-phi) + epsilon^2 D phi,
  Rchi = chi_t + M mu/alpha.
Thus Rphi=alpha D Rchi, while v^T D=0 preserves the original mass exactly.
Every original trajectory admits chi=-(M/alpha) integral(mu,dt); its constant
gauge is auxiliary, not an extra physical boundary condition.

This experimental coordinate changes the loss metric. It does not approximate
or truncate the canonical dense resolvent, and is not a validated default.
"""
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from examples.confined_ch2d.operator_training import apply_operator, encode, time_coordinate
from examples.confined_ch2d.operator_precision import compute_arrays, compute_dtype


def length_squared(c):
    alpha = c.physical.epsilon**2/c.physical.beta
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError('Conservative potential requires positive finite epsilon^2/beta')
    return alpha


def coordinate(net, prepared, t, data, c):
    s = time_coordinate(t, c)
    raw = net.decode(prepared, jnp.reshape(s, (1,)))[0]
    if 'initial_rate_lift' in c.operator:
        tau = data['initial_rate_timescale']
        gate = -jnp.expm1(-t/tau)
        terminal_gate = -jnp.expm1(-c.physical.terminal_time/tau)
        return tau*gate*data['initial_potential_rate']+s*gate/terminal_gate*raw
    return s*raw


def full_state(net, prepared, t, data, c):
    chi, chi_t = jax.jvp(lambda time: coordinate(net, prepared, time, data, c), (t,), (jnp.ones_like(t),))
    alpha = length_squared(c)
    phi = data['initial'] + alpha*apply_operator(chi, data)
    mu = c.physical.beta*(phi**3-phi) + c.physical.epsilon**2*apply_operator(phi, data)
    return phi, mu, c.physical.terminal_time*(chi_t+c.physical.mobility/alpha*mu)


def fields(net, data, times, c):
    data = compute_arrays(data, c)
    times = jnp.asarray(times, dtype=compute_dtype(c))
    prepared = net.prepare_encoded(encode(net, data), data['points'])
    def at(t):
        phi, mu, _ = full_state(net, prepared, t, data, c)
        return jnp.stack((phi, mu), axis=-1)
    return jax.vmap(at)(times)


def compact_payload(data, core, weights, *, inner_capacity, query_capacity):
    """Retain every core->phi-neighbor->potential-neighbor dependency.

    Original points/initial/volumes are encoder context only. Neural queries
    use query_points; no full-grid prediction or mean projection is hidden.
    """
    points, volume, initial = (np.asarray(data[k]) for k in ('points', 'volumes', 'initial'))
    neighbors, coefficients = (np.asarray(data[k]) for k in ('neighbors', 'coefficients'))
    core, weights = np.asarray(core), np.asarray(weights)
    n = len(points)
    if (points.shape != (n, 2) or volume.shape != (n,) or initial.shape != (n,)
            or neighbors.ndim != 2 or neighbors.shape[0] != n or neighbors.shape != coefficients.shape
            or not np.issubdtype(neighbors.dtype, np.integer) or np.any(neighbors < 0) or np.any(neighbors >= n)
            or not all(np.isfinite(x).all() for x in (points, volume, initial, coefficients))
            or np.any(volume < 0) or np.any(coefficients < 0)):
        raise ValueError('Invalid complete physical graph')
    if (core.ndim != 1 or not len(core) or not np.issubdtype(core.dtype, np.integer)
            or weights.shape != core.shape or not np.isfinite(weights).all() or np.any(weights <= 0)
            or np.any(core < 0) or np.any(core >= n) or np.any(volume[core] <= 0)):
        raise ValueError('Core IDs need positive-volume rows and positive finite weights')
    for cap in (inner_capacity, query_capacity):
        if type(cap) is not int or cap < 1:
            raise ValueError('Explicit positive fixed capacities required')
    active_core = coefficients[core] != 0
    selected = np.zeros(n, dtype=bool)
    selected[core] = True
    selected[neighbors[core][active_core]] = True
    inner = np.flatnonzero(selected)
    active_inner = coefficients[inner] != 0
    selected[neighbors[inner][active_inner]] = True
    query = np.flatnonzero(selected)
    if len(inner) > inner_capacity or len(query) > query_capacity:
        raise ValueError(f'Two-hop support overflow: inner{len(inner)}/{inner_capacity}, '
                         f'query{len(query)}/{query_capacity}; no truncated stencil')
    query_map, inner_map = np.full(n, -1), np.full(n, -1)
    query_map[query] = np.arange(len(query))
    inner_map[inner] = np.arange(len(inner))
    padded = np.pad(query, (0, query_capacity-len(query)), mode='edge')
    inner_index = np.pad(query_map[inner], (0, inner_capacity-len(inner)))
    inner_neighbors = query_map[np.where(active_inner, neighbors[inner], query[0])]
    core_neighbors = inner_map[np.where(active_core, neighbors[core], inner[0])]
    if np.any(inner_neighbors < 0) or np.any(core_neighbors < 0):
        raise ValueError('Incomplete dependency closure')
    inner_co = np.pad(coefficients[inner], ((0, inner_capacity-len(inner)), (0, 0)))
    payload = {k: data[k] for k in ('points', 'volumes', 'initial', 'condition', 'condition_weights')}
    payload.update(query_points=points[padded], inner_indices=inner_index.astype(np.int32),
        inner_neighbors=np.pad(inner_neighbors, ((0, inner_capacity-len(inner)), (0, 0))).astype(np.int32),
        inner_coefficients=inner_co, inner_initial=np.pad(initial[inner], (0, inner_capacity-len(inner))),
        core_inner_indices=inner_map[core].astype(np.int32),
        core_query_indices=query_map[core].astype(np.int32),
        core_neighbors=core_neighbors.astype(np.int32), core_coefficients=coefficients[core],
        core_weights=weights.astype(volume.dtype))
    from examples.confined_ch2d.operator_initial_rate import RATE_KEYS
    present = [key in data for key in RATE_KEYS]
    if any(present):
        if not all(present):
            raise ValueError('Both prescribed initial-rate arrays must be present')
        rate, tau = (np.asarray(data[key]) for key in RATE_KEYS)
        if (rate.shape != (n,) or tau.shape != () or not np.isfinite(rate).all()
                or not np.isfinite(tau) or tau <= 0):
            raise ValueError('Invalid initial-rate graph payload')
        payload.update(initial_potential_rate=rate[padded], initial_rate_timescale=tau)
    return payload, dict(core=len(core), inner=len(inner), query=len(query), physical_nodes=int(np.count_nonzero(volume)))


def compact_state(net, prepared, t, data, c):
    chi, chi_t = jax.jvp(lambda time: coordinate(net, prepared, time, data, c), (t,), (jnp.ones_like(t),))
    alpha = length_squared(c)
    first = jnp.sum(data['inner_coefficients']*(chi[data['inner_indices'], None]
                                             -chi[data['inner_neighbors']]), axis=-1)
    phi = data['inner_initial']+alpha*first
    center = phi[data['core_inner_indices']]
    second = jnp.sum(data['core_coefficients']*(center[:, None]-phi[data['core_neighbors']]), axis=-1)
    mu = c.physical.beta*(center**3-center)+c.physical.epsilon**2*second
    residual = c.physical.terminal_time*(chi_t[data['core_query_indices']]+c.physical.mobility/alpha*mu)
    return center, mu, residual


def make_loss(c):
    """Same physical time measure; local potential residual replaces dense norm."""
    length_squared(c)
    def one(net, data, times):
        compact = 'core_query_indices' in data
        queries = data['query_points'] if compact else data['points']
        prepared = net.prepare_encoded(encode(net, data), queries)
        state = compact_state if compact else full_state
        weights = data['core_weights'] if compact else data['volumes']/jnp.sum(data['volumes'])
        values = jax.vmap(lambda t: jnp.dot(weights, state(net, prepared, t, data, c)[2]**2))(times)
        if 'initial_residual_fraction' in c.loss:
            values = eqx.error_if(values, times[0] != 0., 'Initial residual fraction requires physical t=0')
            fraction = c.loss.initial_residual_fraction
            return fraction*values[0]+(1-fraction)*jnp.mean(values[1:])
        return jnp.mean(values)
    def loss(net, batch, times):
        batch = compute_arrays(batch, c)
        times = jnp.asarray(times, dtype=compute_dtype(c))
        return jnp.mean(jax.vmap(one, in_axes=(None, 0, None))(net, batch, times))
    return loss
