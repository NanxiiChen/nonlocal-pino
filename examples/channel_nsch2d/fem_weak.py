"""Mixed P2/P1 FE weak NS--CH rows on the original full query grid.

NumPy geometry construction has no JAX/FEniCS dependency. The weak form follows
solver.py: symmetric stress, conservative phase transport and outlet phase flux.
P2 row-sum masses are not positive; use the consistent mass diagonal instead.
"""
import numpy as np


INDEX_KEYS = ('fem_cells', 'fem_assembly', 'fem_outlet')
PARTS = ('phase', 'chemical', 'momentum_x', 'momentum_y')


def validate(c):
    spec = dict(method='fem_weak_p2p1', quadrature_order=6,
                residual_metric='mass_diagonal_dual', geometry='isoparametric_p2')
    if c.nl.to_dict() != spec:
        raise ValueError(f'FE control requires explicit nl={spec}')
    if (c.sample.Nx % 2 or c.sample.Ny % 2 or min(c.sample.Nx, c.sample.Ny) < 4
            or 'velocity' not in c or c.velocity.kind != 'streamfunction'
            or 'time_derivative' not in c
            or c.time_derivative.to_dict() != dict(kind='jvp', sampling='continuous_stratified')
            or 'chemical_residual_scale' not in c.loss):
        raise ValueError('P2/P1 control requires an even full grid, streamfunction, JVP and explicit chemical scale')
    excluded = ('derivatives', 'phase_transport', 'balance', 'causal', 'stagger',
                'chemical_constraint', 'phase_coordinate', 'residual_transform',
                'stress_formulation', 'pressure_coordinate', 'phase_metric', 'droplet_metric')
    if any(k in c for k in excluded) or 'spatial' in c.sample:
        raise ValueError('First FE control requires full coordinates and original joint physical outputs')
    if any(k in c.train for k in ('continuation', 'refinement', 'initialization')):
        raise ValueError('FE control must start cold')


def triangle_rule(order):
    roots, weights = np.polynomial.legendre.leggauss(order)
    s, t = np.meshgrid((roots + 1) / 2, (roots + 1) / 2, indexing='ij')
    w = weights[:, None] * weights[None, :] / 4 * (1 - s)
    return np.column_stack((s.ravel(), ((1 - s) * t).ravel())), w.ravel()


def basis(points):
    """Lagrange P2 basis: vertices 0,1,2 then edges 01,12,20."""
    x, y = np.asarray(points).T
    lam = np.column_stack((1 - x - y, x, y))
    dl = np.array([[-1., -1.], [1., 0.], [0., 1.]])
    values = [lam[:, i] * (2 * lam[:, i] - 1) for i in range(3)]
    grads = [(4 * lam[:, i, None] - 1) * dl[i] for i in range(3)]
    for i, j in ((0, 1), (1, 2), (2, 0)):
        values.append(4 * lam[:, i] * lam[:, j])
        grads.append(4 * (lam[:, i, None] * dl[j] + lam[:, j, None] * dl[i]))
    return np.stack(values, axis=1), np.stack(grads, axis=1), lam


def connectivity(nx, ny):
    """Two P2 triangles per 2x2 fine-grid cell, with all original nodes used."""
    ix, iy = np.meshgrid(np.arange(0, nx, 2), np.arange(0, ny, 2), indexing='ij')
    base = np.column_stack((ix.ravel(), iy.ravel()))
    patterns = np.array([[[0, 0], [2, 0], [2, 2], [1, 0], [2, 1], [1, 1]],
                         [[0, 0], [2, 2], [0, 2], [1, 1], [1, 2], [0, 1]]])
    xy = (base[:, None, None, :] + patterns[None]).reshape(-1, 6, 2)
    cells = (xy[..., 0] * (ny + 1) + xy[..., 1]).astype(np.int32)
    n = (nx + 1) * (ny + 1)
    flat = cells.ravel()
    counts = np.bincount(flat, minlength=n)
    if counts.min() < 1 or counts.max() > 6:
        raise ValueError('P2 mesh requires complete support with at most six incident triangles')
    order = np.argsort(flat, kind='stable')
    starts = np.r_[0, np.cumsum(counts)[:-1]]
    rank = np.arange(len(flat)) - np.repeat(starts, counts)
    assembly = np.full((n, 6), len(flat), dtype=np.int32)
    assembly[flat[order], rank] = order
    y = np.arange(0, ny, 2)
    outlet = (nx * (ny + 1) + np.column_stack((y, y + 2, y + 1))).astype(np.int32)
    return cells, assembly, outlet


def build(points, c):
    validate(c)
    points = np.asarray(points, dtype=np.float64)
    nx, ny = c.sample.Nx, c.sample.Ny
    n = (nx + 1) * (ny + 1)
    if points.shape != (n, 2) or not np.isfinite(points).all():
        raise ValueError('Finite complete mapped tensor coordinates required')
    expected_x=np.repeat(np.linspace(0,c.geometry.length,nx+1),ny+1)
    if not np.array_equal(points[:,0],expected_x):
        raise ValueError('FE control requires the declared tensor x-coordinate ordering')
    cells, assembly, outlet = connectivity(nx, ny)
    q, w = triangle_rule(c.nl.quadrature_order)
    shape, gradient, _ = basis(q)
    jac = np.einsum('eia,qib->eqab', points[cells], gradient)
    determinant = np.linalg.det(jac)
    if not np.isfinite(determinant).all() or np.any(determinant <= 0):
        raise ValueError('Invalid quadratic FE geometry: nonpositive Jacobian')
    # x is affine for this map, so det(J) is linear on each triangle. Its
    # vertex extrema also certify positivity between quadrature points.
    _,vertex_gradient,_=basis(np.array([[0.,0.],[1.,0.],[0.,1.]]))
    vertex_det=np.linalg.det(np.einsum('eia,qib->eqab',points[cells],vertex_gradient))
    if np.any(vertex_det<=0):
        raise ValueError('Quadratic FE geometry has a nonpositive vertex Jacobian')
    physical_gradient = np.einsum('qib,eqba->eqia', gradient, np.linalg.inv(jac))
    measure = determinant * w[None, :]
    diagonal = np.zeros(n)
    element_diagonal = np.einsum('eq,qi->ei', measure, shape**2)
    np.add.at(diagonal, cells.ravel(), element_diagonal.ravel())
    if np.any(diagonal <= 0) or not np.isfinite(diagonal).all():
        raise ValueError('Consistent P2 mass diagonal must be positive')
    z, zw = np.polynomial.legendre.leggauss(c.nl.quadrature_order)
    s = (z + 1) / 2
    edge_grad = np.column_stack((4*s - 3, 4*s - 1, 4 - 8*s))
    tangent = np.einsum('qi,eia->eqa', edge_grad, points[outlet])
    edge_measure = np.linalg.norm(tangent, axis=-1) * zw[None, :] / 2
    # Outward normal times ds: outlet edges are oriented bottom to top.
    normal_measure = np.stack((tangent[..., 1], -tangent[..., 0]), axis=-1) * zw[None, :, None] / 2
    active = np.ones((nx + 1, ny + 1, 4))
    active[0] = 0                       # inlet test traces vanish
    active[:, [0, -1], 2:] = 0          # no-slip velocity test traces vanish
    data = dict(fem_cells=cells, fem_assembly=assembly, fem_outlet=outlet,
                fem_gradient=physical_gradient, fem_measure=measure,
                fem_mass_diagonal=diagonal, fem_active=active.reshape(n, 4),
                fem_outlet_normal_measure=normal_measure)
    report = dict(method=c.nl.method, nodes=n, triangles=len(cells),
                  geometry='P2 isoparametric on unchanged mapped query nodes',
                  field_order='P2(phi,mu,u), P1(p); physical velocity remains AD curl',
                  quadrature_points=len(q), area=float(measure.sum()),
                  minimum_jacobian=float(min(determinant.min(),vertex_det.min())),
                  minimum_mass_diagonal=float(diagonal.min()),
                  outlet_length=float(edge_measure.sum()),
                  boundary='Natural phase/mu flux and traction; explicit conservative outlet transport',
                  residual_metric=c.nl.residual_metric, pde_solves=0)
    return data, report


def shapes(c):
    n = (c.sample.Nx + 1) * (c.sample.Ny + 1)
    e = c.sample.Nx * c.sample.Ny // 2
    q = c.nl.quadrature_order**2
    from examples.channel_nsch2d.geometry import parameter_count
    from examples.channel_nsch2d.problem import boundary_payload_shapes
    b = 4 * (c.boundary_sampling.points_per_side if 'boundary_sampling' in c else c.sample.boundary_points)
    result = dict(points=(n, 2), volumes=(n,), interior=(n,),
        amplitudes=(parameter_count(c),), boundary=(b, 5), boundary_weights=(b,),
        domain=(n, 3), domain_weights=(n,), fem_cells=(e, 6), fem_assembly=(n, 6),
        fem_outlet=(c.sample.Ny//2, 3), fem_gradient=(e, q, 6, 2),
        fem_measure=(e, q), fem_mass_diagonal=(n,), fem_active=(n, 4),
        fem_outlet_normal_measure=(c.sample.Ny//2, c.nl.quadrature_order, 2))
    result.update(boundary_payload_shapes(c))
    return result


def geometry_data(amplitudes, c):
    """Keep both encoder streams identical to the existing nonlocal control."""
    from examples.channel_nsch2d.geometry import validate_parameters
    from examples.channel_nsch2d.problem import quadrature, boundary, boundary_loss_quadrature, initial_field
    a = validate_parameters(amplitudes, c)
    points, volumes = quadrature(a, c, c.sample.Nx, c.sample.Ny)
    b, bw = boundary(a, c)
    interior = np.ones((c.sample.Nx+1, c.sample.Ny+1))
    interior[[0, -1], :] = 0
    interior[:, [0, -1]] = 0
    data = dict(points=points, volumes=volumes, interior=interior.ravel(),
                amplitudes=a, boundary=b, boundary_weights=bw,
                domain=np.column_stack((points, initial_field(points, c, xp=np))), domain_weights=volumes)
    if 'boundary_sampling' in c:
        data.update(encoder_boundary=b, encoder_boundary_weights=bw)
        b, bw = boundary_loss_quadrature(a, c)
        data.update(boundary=b, boundary_weights=bw)
    finite_elements, report = build(points, c)
    data.update(finite_elements)
    return data, report


def validate_payload(data, c):
    expected = shapes(c)
    if set(data) != set(expected):
        raise ValueError('FE geometry payload must contain only declared known-input arrays')
    for key, value in data.items():
        dtype = np.dtype('int32' if key in INDEX_KEYS else c.train.compute_dtype)
        if value.shape != expected[key] or value.dtype != dtype or not np.isfinite(value).all():
            raise ValueError(f'Invalid FE payload {key}: {value.shape}/{value.dtype}, expected {expected[key]}/{dtype}')
    n = len(data['points'])
    for key in ('fem_cells', 'fem_outlet'):
        if np.any(data[key] < 0) or np.any(data[key] >= n):
            raise ValueError(f'Out-of-range FE connectivity: {key}')
    if np.any(data['fem_assembly'] < 0) or np.any(data['fem_assembly'] > data['fem_cells'].size):
        raise ValueError('Out-of-range FE assembly map')
    for key in ('volumes', 'boundary_weights', 'domain_weights', 'fem_measure', 'fem_mass_diagonal'):
        if np.any(data[key] <= 0):
            raise ValueError(f'Nonpositive FE measure: {key}')
    if not np.all(np.isin(data['fem_active'], [0, 1])):
        raise ValueError('FE test masks must be binary')
    np.testing.assert_array_equal(data['domain'][:, :2], data['points'])
    np.testing.assert_array_equal(data['domain_weights'], data['volumes'])


def make_weak_rows(c):
    """Element quadrature and deterministic assembly; no network spatial AD here."""
    import jax
    import jax.numpy as jnp
    from examples.channel_nsch2d.problem import material
    q, _ = triangle_rule(c.nl.quadrature_order)
    N, _, P = basis(q)
    N, P = jnp.asarray(N, dtype=c.train.compute_dtype), jnp.asarray(P, dtype=c.train.compute_dtype)
    roots, _ = np.polynomial.legendre.leggauss(c.nl.quadrature_order)
    s = (roots + 1) / 2
    E = jnp.asarray(np.column_stack(((1-s)*(1-2*s), s*(2*s-1), 4*s*(1-s))), dtype=c.train.compute_dtype)
    A, kappa = material(c)

    def assemble(local, incidence):
        flat = local.reshape(local.shape[0], -1, local.shape[-1])
        padded = jnp.concatenate((flat, jnp.zeros_like(flat[:, :1])), axis=1)
        return jnp.sum(padded[:, incidence, :], axis=2)

    @jax.custom_vjp
    def gather(values, cells, incidence):
        return values[:, cells, :]

    def gather_fwd(values, cells, incidence):
        return values[:, cells, :], (incidence,)

    def gather_bwd(saved, cotangent):
        return assemble(cotangent, saved[0]), None, None

    gather.defvjp(gather_fwd, gather_bwd)

    @jax.custom_vjp
    def scatter(local, cells, incidence):
        return assemble(local, incidence)

    def scatter_fwd(local, cells, incidence):
        return assemble(local, incidence), (cells,)

    def scatter_bwd(saved, cotangent):
        return cotangent[:, saved[0], :], None, None

    scatter.defvjp(scatter_fwd, scatter_bwd)

    def rows(value, dt, data):
        cells, incidence = data['fem_cells'], data['fem_assembly']
        v = gather(value, cells, incidence)
        rate = gather(dt, cells, incidence)
        z = jnp.einsum('qi,teic->teqc', N, v)
        zt = jnp.einsum('qi,teic->teqc', N, rate)
        grad = jnp.einsum('eqia,teic->teqca', data['fem_gradient'], v)
        pressure = jnp.einsum('qi,tei->teq', P, v[:, :, :3, 4])
        phi, mu, ux, uy = [z[..., i] for i in range(4)]
        rho, eta, mobility = c.physical.density, c.physical.viscosity, c.physical.mobility
        force_x = phi * grad[..., 1, 0]
        force_y = phi * grad[..., 1, 1]
        adv_x = ux * grad[..., 2, 0] + uy * grad[..., 2, 1]
        adv_y = ux * grad[..., 3, 0] + uy * grad[..., 3, 1]
        load = jnp.stack((zt[..., 0], mu - A*(phi**3-phi),
                          rho*(zt[..., 2]+adv_x)+force_x,
                          rho*(zt[..., 3]+adv_y)+force_y), axis=-1)
        shear = eta*(grad[..., 2, 1]+grad[..., 3, 0])
        flux_x = jnp.stack((-phi*ux+mobility*grad[..., 1, 0],
                            -kappa*grad[..., 0, 0], 2*eta*grad[..., 2, 0]-pressure, shear), axis=-1)
        flux_y = jnp.stack((-phi*uy+mobility*grad[..., 1, 1],
                            -kappa*grad[..., 0, 1], shear, 2*eta*grad[..., 3, 1]-pressure), axis=-1)
        w, g = data['fem_measure'], data['fem_gradient']
        local = (jnp.einsum('eq,qi,teqc->teic', w, N, load)
                 + jnp.einsum('eq,eqi,teqc->teic', w, g[..., 0], flux_x)
                 + jnp.einsum('eq,eqi,teqc->teic', w, g[..., 1], flux_y))
        result = scatter(local, cells, incidence)
        edge = jnp.einsum('qi,teic->teqc', E, value[:, data['fem_outlet'], :])
        transport = edge[..., 0] * jnp.sum(edge[..., 2:4]*data['fem_outlet_normal_measure'][None], axis=-1)
        boundary_load = jnp.einsum('qi,teq->tei', E, transport)
        vertices=jnp.concatenate((boundary_load[:,:1,0],
            boundary_load[:,1:,0]+boundary_load[:,:-1,1],boundary_load[:,-1:,1]),axis=1)
        trace=jnp.concatenate((jnp.stack((vertices[:,:-1],boundary_load[:,:,2]),axis=-1).reshape(len(value),-1),
                               vertices[:,-1:]),axis=1)
        result = result.at[:, -(c.sample.Ny+1):, 0].add(trace)
        return result
    return rows


def make_loss_fn(c):
    import jax
    import jax.numpy as jnp
    from examples.channel_nsch2d.network import encode, time_jets
    rows = make_weak_rows(c)
    scales = jnp.asarray([c.loss.phase_scale, c.loss.chemical_residual_scale,
                          c.loss.momentum_scale, c.loss.momentum_scale], dtype=c.train.compute_dtype)

    def one(net, data, times):
        value, dt = time_jets(net, encode(net, data), data['points'], times, data['amplitudes'], c)
        residual = rows(value, dt, data) / scales
        return jnp.sum(residual**2 * data['fem_active'][None] /
                       data['fem_mass_diagonal'][None, :, None], axis=(0, 1)) / (len(times)*jnp.sum(data['fem_measure']))

    def loss(net, batch, times):
        parts = jnp.mean(jax.vmap(lambda data: one(net, data, times))(batch), axis=0)
        return jnp.sum(parts), parts
    return loss
