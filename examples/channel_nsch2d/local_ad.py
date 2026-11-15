"""Full local spatial AD, with the original channel quadrature and BC losses.

Each query is independent conditional on the encoded geometry. Directional
JVPs across all queries therefore give pointwise derivatives, not derivatives
of the geometry tokens. Velocity is already the AD curl of a streamfunction;
its Laplacian here includes third spatial derivatives of that potential.
"""
import numpy as np


def validate(c):
    spec = dict(method='local_ad', loss_quadrature='sbp4_mapped')
    if c.nl.to_dict() != spec:
        raise ValueError(f'Local AD control requires explicit nl={spec}')
    if (min(c.sample.Nx, c.sample.Ny) < 8 or 'velocity' not in c
            or c.velocity.kind != 'streamfunction' or 'time_derivative' not in c
            or c.time_derivative.to_dict() != dict(kind='jvp', sampling='continuous_stratified')
            or 'chemical_residual_scale' not in c.loss):
        raise ValueError('Local AD requires full mapped grid, streamfunction, time JVP and chemical scale')
    excluded = ('derivatives', 'phase_transport', 'balance', 'causal', 'stagger',
                'chemical_constraint', 'phase_coordinate', 'residual_transform',
                'stress_formulation', 'pressure_coordinate', 'phase_metric', 'droplet_metric')
    if any(k in c for k in excluded) or 'spatial' in c.sample:
        raise ValueError('First local AD control requires full coordinates and original joint outputs')
    if any(k in c.train for k in ('continuation', 'refinement', 'initialization')):
        raise ValueError('Local AD control must start cold')
    if 'ad_sampling' in c.sample:
        s = c.sample.ad_sampling
        count = (c.sample.Nx-1)*(c.sample.Ny-1)
        if (set(s) != {'kind', 'core_count', 'replacement'} or s.kind != 'uniform_interior'
                or s.replacement is not False or isinstance(s.core_count, bool)
                or not isinstance(s.core_count, int) or not 1 <= s.core_count <= count):
            raise ValueError('AD sampling requires explicit uniform_interior, valid core_count and replacement=False')


def shapes(c):
    from examples.channel_nsch2d.geometry import parameter_count
    from examples.channel_nsch2d.problem import boundary_payload_shapes
    n = (c.sample.Nx+1)*(c.sample.Ny+1)
    b = 4*(c.boundary_sampling.points_per_side if 'boundary_sampling' in c else c.sample.boundary_points)
    result = dict(points=(n, 2), volumes=(n,), interior=(n,),
        amplitudes=(parameter_count(c),), boundary=(b, 5), boundary_weights=(b,),
        domain=(n, 3), domain_weights=(n,), transport_measure=(n,))
    result.update(boundary_payload_shapes(c))
    return result


def geometry_data(amplitudes, c):
    """Known conditions and loss quadrature only; no field derivative stencils."""
    from examples.channel_nsch2d.geometry import validate_parameters
    from examples.channel_nsch2d.problem import quadrature, boundary, boundary_loss_quadrature, initial_field
    from examples.channel_nsch2d.transport import build_transport
    validate(c)
    a = validate_parameters(amplitudes, c)
    points, volumes = quadrature(a, c, c.sample.Nx, c.sample.Ny)
    b, bw = boundary(a, c)
    interior = np.ones((c.sample.Nx+1, c.sample.Ny+1))
    interior[[0, -1], :] = 0
    interior[:, [0, -1]] = 0
    # Reuse the baseline's checked physical SBP norm; none of its derivative
    # matrices is retained or applied to neural fields in this AD route.
    measure = build_transport(points, a, c)['transport_measure']
    data = dict(points=points, volumes=volumes, interior=interior.ravel(),
        amplitudes=a, boundary=b, boundary_weights=bw,
        domain=np.column_stack((points, initial_field(points, c, xp=np))),
        domain_weights=volumes, transport_measure=measure)
    if 'boundary_sampling' in c:
        data.update(encoder_boundary=b, encoder_boundary_weights=bw)
        b, bw = boundary_loss_quadrature(a, c)
        data.update(boundary=b, boundary_weights=bw)
    return data, dict(method='local_ad', nodes=len(points), area=float(measure.sum()),
        pde_solves=0, field_derivative_stencils=0,
        scope='Spatial AD for all PDE/BC derivatives; same mapped loss norm and encoder inputs')


def validate_payload(data, c):
    from examples.channel_nsch2d.geometry import coefficient_bounds
    from examples.channel_nsch2d.problem import boundary, boundary_loss_quadrature, initial_field
    expected = shapes(c)
    if set(data) != set(expected):
        raise ValueError('Local AD payload requires only known-input geometry and quadrature arrays')
    dtype = np.dtype(c.train.compute_dtype)
    for key, value in data.items():
        if value.shape != expected[key] or value.dtype != dtype or not np.isfinite(value).all():
            raise ValueError(f'Invalid local AD {key}: {value.shape}/{value.dtype}, expected {expected[key]}/{dtype}')
    for key in ('volumes', 'domain_weights', 'boundary_weights', 'transport_measure'):
        if np.any(data[key] <= 0):
            raise ValueError(f'Nonpositive local AD quadrature: {key}')
    interior = np.ones((c.sample.Nx+1, c.sample.Ny+1))
    interior[[0, -1], :] = 0
    interior[:, [0, -1]] = 0
    np.testing.assert_array_equal(data['interior'], interior.ravel())
    np.testing.assert_array_equal(data['domain'][:, :2], data['points'])
    np.testing.assert_array_equal(data['domain_weights'], data['volumes'])
    bounds = np.asarray(coefficient_bounds(c), dtype=dtype)
    if np.any(data['amplitudes'] < bounds[0]) or np.any(data['amplitudes'] > bounds[1]):
        raise ValueError('Local AD wall amplitudes outside declared prior')
    eps = np.finfo(dtype).eps
    np.testing.assert_allclose(data['domain'][:, 2], initial_field(data['points'], c, xp=np), atol=32*eps, rtol=32*eps)
    checks = [(('boundary', 'boundary_weights'), boundary)]
    if 'boundary_sampling' in c:
        checks = [(('encoder_boundary', 'encoder_boundary_weights'), boundary),
                  (('boundary', 'boundary_weights'), boundary_loss_quadrature)]
    for names, fn in checks:
        for name, value in zip(names, fn(data['amplitudes'], c), strict=True):
            np.testing.assert_allclose(data[name], value, atol=8*eps, rtol=8*eps, err_msg=name)


def spatial_jets(fn, points):
    """Pointwise value, dx, dy, Laplacian for a query-independent field fn."""
    import jax
    import jax.numpy as jnp
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError('Local AD requires (N,2) physical query coordinates')
    ex = jnp.broadcast_to(jnp.array([1., 0.], dtype=points.dtype), points.shape)
    ey = ex[:, ::-1]
    value, dx = jax.jvp(fn, (points,), (ex,))
    _, dxx = jax.jvp(lambda x: jax.jvp(fn, (x,), (ex,))[1], (points,), (ex,))
    dy, dyy = jax.jvp(lambda x: jax.jvp(fn, (x,), (ey,))[1], (points,), (ey,))
    return value, dx, dy, dxx+dyy


def conservative_advection(value, dx, dy):
    """Exact product rule for div(phi*v), retaining even numerical divergence."""
    return (value[..., 2]*dx[..., 0] + value[..., 3]*dy[..., 0]
            + value[..., 0]*(dx[..., 2]+dy[..., 3]))


QUERY_NAMESPACE = 271828


def query_indices(data_key, step, c):
    """Uniform interior subsets per geometry occurrence, independent of IC/time RNG."""
    import jax
    import jax.numpy as jnp
    key = jax.random.fold_in(jax.random.fold_in(data_key, QUERY_NAMESPACE), step)
    keys = jax.random.split(key, c.operator.condition_batch)
    nx, ny = c.sample.Nx-1, c.sample.Ny-1
    selected = jax.vmap(lambda k: jax.random.choice(
        k, nx*ny, shape=(c.sample.ad_sampling.core_count,), replace=False))(keys)
    return ((selected//ny+1)*(c.sample.Ny+1)+(selected % ny+1)).astype(jnp.int32)


def sample_batch(batch, indices, c):
    """Unbiased full-quadrature estimator; never renormalize sampled weights.

    Each of M interior nodes has inclusion probability K/M. The full loss is
    sum_i m_i R_i^2 / W. Averaging M*m_i/W * R_i^2 over K selected nodes has
    exactly this expectation. BC quadrature and encoder tokens are untouched.
    """
    import jax.numpy as jnp
    shape = (c.operator.condition_batch, c.sample.ad_sampling.core_count)
    if indices.shape != shape:
        raise ValueError(f'AD query indices must have shape {shape}')
    measure = batch['transport_measure']*batch['interior']
    selected = jnp.take_along_axis(measure, indices, axis=1)
    count = (c.sample.Nx-1)*(c.sample.Ny-1)
    weights = count*selected/jnp.sum(measure, axis=1, keepdims=True)
    return dict(batch, ad_core_indices=indices, ad_core_weights=weights)


def audit_queries(saved_indices, data_key, c):
    """Replay every actual sampled subset in bounded chunks after training."""
    import jax
    import jax.numpy as jnp
    shape = (c.train.iterations, c.operator.condition_batch, c.sample.ad_sampling.core_count)
    if saved_indices.shape != shape or saved_indices.dtype != np.dtype('int32'):
        raise ValueError(f'Actual AD subsets must have shape {shape} and dtype int32')
    replay = jax.jit(jax.vmap(lambda step: query_indices(data_key, step, c)))
    for begin in range(0, len(saved_indices), 16):
        end = min(begin+16, len(saved_indices))
        steps = np.minimum(np.arange(begin, begin+16), len(saved_indices)-1).astype(np.uint32)
        expected = np.asarray(replay(jnp.asarray(steps)))[:end-begin]
        actual = saved_indices[begin:end]
        np.testing.assert_array_equal(actual, expected)
        ix, iy = actual//(c.sample.Ny+1), actual % (c.sample.Ny+1)
        if not np.all((ix > 0) & (ix < c.sample.Nx) & (iy > 0) & (iy < c.sample.Ny)):
            raise ValueError('AD query is not strictly interior')
        if np.any(np.diff(np.sort(actual, axis=-1), axis=-1) == 0):
            raise ValueError('Uniform AD subsets must not contain duplicates')
    return dict(status='passed', actual_subsets=int(np.prod(shape[:2])), queries_per_subset=shape[2],
                independent_boundary_quadrature=True, all_query_draws_replayed=True)
