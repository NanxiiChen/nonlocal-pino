"""Full-field local-AD CH with exact known IC jets and physical wall losses.

The original nonlocal stable resolvent is retained only as a frozen loss metric.
PDE/BC derivatives below contain no graph differentiation. Global mass means
depend on time and parameters but are constants with respect to query position.
"""
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from examples.confined_ch2d.operator_precision import compute_arrays, compute_dtype


def validate_local_ad(c):
    spec = c.operator.local_ad
    expected = {'query_chunk', 'boundary', 'phase_flux_scale', 'chemical_flux_scale', 'boundary_weight'}
    if set(spec) != expected or spec.boundary != 'existing_physical_measure_positive_times':
        raise ValueError('AD requires explicit chunk and two physical Neumann normalizers/weight')
    if (type(spec.query_chunk) is not int or spec.query_chunk < 1
            or c.operator.metric != 'linear_stable_resolvent'
            or 'formulation' in c.operator or 'spatial' in c.sample or 'residual_rows' in c.sample
            or 'phase' in c or 'initial_rhs_condition' in c.operator or 'augmentation' in c.operator
            or 'initial_residual_fraction' not in c.loss):
        raise ValueError('AD comparison preserves the original full-field linear-time mass-projected scheme')
    scales = [spec.phase_flux_scale, spec.chemical_flux_scale, spec.boundary_weight]
    if not np.isfinite(scales).all() or min(scales) <= 0:
        raise ValueError('Positive finite physical boundary scales and weight required')
    if c.operator.online.max_points % spec.query_chunk:
        raise ValueError('AD query chunk must divide the explicit padded full-field point count')


def initial_jets(points, c):
    """Exact value, gradient, Laplacian, bi-Laplacian and gradient of Laplacian."""
    waves = jnp.asarray(c.initial.waves, dtype=points.dtype)
    k = jnp.pi*waves[:, :2]/jnp.asarray(c.geometry.axes, dtype=points.dtype)
    angle = (points-jnp.asarray(c.geometry.center, dtype=points.dtype)) @ k.T+waves[:, 3]
    amplitude = c.initial.amplitude*waves[:, 2]
    cosine, sine = amplitude*jnp.cos(angle), amplitude*jnp.sin(angle)
    k2 = jnp.sum(k*k, axis=1)
    return (jnp.sum(cosine, axis=1), -sine @ k,
            -(cosine @ k2), cosine @ (k2*k2), (sine*k2) @ k)


def raw(net, encoded, points, t, c):
    prepared = net.prepare_encoded(encoded, points)
    return net.decode(prepared, jnp.reshape(t/c.physical.terminal_time, (1,)))[0]


def directional(function, points, direction):
    tangent = jnp.broadcast_to(direction, points.shape)
    return jax.jvp(function, (points,), (tangent,))[1]


def spatial_jets(function, points, fourth):
    ex, ey = jnp.eye(2, dtype=points.dtype)
    dx = lambda x: directional(function, x, ex)
    dy = lambda x: directional(function, x, ey)
    dxx = lambda x: directional(dx, x, ex)
    dyy = lambda x: directional(dy, x, ey)
    gradient = jnp.stack((dx(points), dy(points)), axis=-1)
    laplacian = dxx(points)+dyy(points)
    if not fourth:
        return gradient, laplacian
    xxxx = directional(lambda x: directional(dxx, x, ex), points, ex)
    xxyy = directional(lambda x: directional(dxx, x, ey), points, ey)
    yyyy = directional(lambda x: directional(dyy, x, ey), points, ey)
    return gradient, laplacian, xxxx+2*xxyy+yyyy


def chunked_jets(net, encoded, points, t, c, fourth):
    count = c.operator.local_ad.query_chunk
    if len(points) % count:
        raise ValueError('Full AD point padding must divide query_chunk')

    @eqx.filter_checkpoint
    def block(x):
        return spatial_jets(lambda xx: raw(net, encoded, xx, t, c), x, fourth)

    values = jax.lax.map(block, points.reshape((-1, count, 2)))
    return tuple(value.reshape((len(points),)+value.shape[2:]) for value in values)


def one_time(net, data, encoded, prepared, t, c):
    from examples.confined_ch2d.operator_training import phase
    phi, dt = jax.jvp(lambda tt: phase(net, prepared, tt, data, c), (t,), (jnp.ones_like(t),))
    _, g0, l0, b0, _ = initial_jets(data['points'], c)
    gf, lf, bf = chunked_jets(net, encoded, data['points'], t, c, True)
    s = t/c.physical.terminal_time
    gradient, laplacian, biharmonic = g0+s*gf, l0+s*lf, b0+s*bf
    p = c.physical
    lap_mu = p.beta*((3*phi**2-1)*laplacian+6*phi*jnp.sum(gradient**2, axis=-1))-p.epsilon**2*biharmonic
    residual = p.terminal_time*(dt-p.mobility*lap_mu)
    filtered = data['metric'] @ residual
    pde = jnp.dot(data['volumes'], filtered**2)/jnp.sum(data['volumes'])
    x, n, w = data['condition'][:, :2], data['condition'][:, 2:4], data['condition_weights']
    value0, gradient0, _, _, lap_gradient0 = initial_jets(x, c)
    at = lambda xx: raw(net, encoded, xx, t, c)
    fb, fn = jax.jvp(at, (x,), (n,))
    lap_fn = lambda xx: spatial_jets(at, xx, False)[1]
    lap_normal = jax.jvp(lap_fn, (x,), (n,))[1]
    change = net.decode(prepared, jnp.reshape(s, (1,)))[0]
    mean = jnp.dot(data['volumes'], change)/jnp.sum(data['volumes'])
    phi_b = value0+s*(fb-mean)
    phi_n = jnp.sum(gradient0*n, axis=-1)+s*fn
    lap_phi_n = jnp.sum(lap_gradient0*n, axis=-1)+s*lap_normal
    mu_n = p.beta*(3*phi_b**2-1)*phi_n-p.epsilon**2*lap_phi_n
    bc_phi = jnp.dot(w, (phi_n/c.operator.local_ad.phase_flux_scale)**2)/jnp.sum(w)
    bc_mu = jnp.dot(w, (mu_n/c.operator.local_ad.chemical_flux_scale)**2)/jnp.sum(w)
    return jnp.stack((pde, bc_phi, bc_mu))


def condition_parts(net, data, times, c):
    from examples.confined_ch2d.operator_training import encode
    times = eqx.error_if(times, times[0] != 0, 'AD time stream requires its leading initial coordinate')
    encoded = encode(net, data)
    prepared = net.prepare_encoded(encoded, data['points'])
    fraction = c.loss.initial_residual_fraction
    if fraction == 0.:
        return jax.vmap(lambda t: one_time(net, data, encoded, prepared, t, c))(times[1:]).mean(axis=0)
    parts = jax.vmap(lambda t: one_time(net, data, encoded, prepared, t, c))(times)
    pde = fraction*parts[0, 0]+(1-fraction)*parts[1:, 0].mean()
    # The hard initial ambient function is incompatible with new wall normals.
    # Preserve it; the soft BC is imposed for positive times only.
    return jnp.concatenate((pde[None], parts[1:, 1:].mean(axis=0)))


def make_loss(c):
    validate_local_ad(c)

    def loss(net, batch, times):
        batch, times = compute_arrays(batch, c), jnp.asarray(times, dtype=compute_dtype(c))
        parts = jax.vmap(lambda data: condition_parts(net, data, times, c))(batch).mean(axis=0)
        return parts[0]+c.operator.local_ad.boundary_weight*jnp.sum(parts[1:])
    return loss


def make_step(c, optimizer):
    """Average complete geometry gradients, then clip/update SOAP exactly once."""
    validate_local_ad(c)

    def objective(net, data, times):
        parts = condition_parts(net, data, times, c)
        return parts[0]+c.operator.local_ad.boundary_weight*jnp.sum(parts[1:]), parts

    @eqx.filter_jit
    def step(net, state, batch, times):
        batch, times = compute_arrays(batch, c), jnp.asarray(times, dtype=compute_dtype(c))
        count = len(batch['points'])
        gradient0 = jax.tree.map(jnp.zeros_like, eqx.filter(net, eqx.is_array))

        def add(i, totals):
            value, gradient, parts = totals
            data = jax.tree.map(lambda a: a[i], batch)
            (vi, pi), gi = eqx.filter_value_and_grad(objective, has_aux=True)(net, data, times)
            return value+vi/count, jax.tree.map(lambda a, b: a+b/count, gradient, gi), parts+pi/count

        value, gradient, parts = jax.lax.fori_loop(0, count, add,
            (jnp.asarray(0., dtype=compute_dtype(c)), gradient0, jnp.zeros(3, dtype=compute_dtype(c))))
        finite = jnp.isfinite(value) & jnp.all(jnp.stack([
            jnp.all(jnp.isfinite(a)) for a in jax.tree.leaves(gradient) if eqx.is_array(a)]))
        updates, state = optimizer.update(gradient, state, net)
        return eqx.apply_updates(net, updates), state, value, finite, parts
    return step


def fields(net, data, times, c):
    from examples.confined_ch2d.operator_training import encode, phase
    data, times = compute_arrays(data, c), jnp.asarray(times, dtype=compute_dtype(c))
    encoded = encode(net, data)
    prepared = net.prepare_encoded(encoded, data['points'])
    l0 = initial_jets(data['points'], c)[2]

    def at(t):
        phi = phase(net, prepared, t, data, c)
        lf = chunked_jets(net, encoded, data['points'], t, c, False)[1]
        mu = c.physical.beta*(phi**3-phi)-c.physical.epsilon**2*(l0+t/c.physical.terminal_time*lf)
        return jnp.stack((phi, jnp.where(data['volumes'] > 0, mu, 0.)), axis=-1)
    return jax.vmap(at)(times)
