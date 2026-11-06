"""Sample fixed impermeable inclusions, without PDE data or learned outputs.

Each contour is an invertible affine image of a positive smooth radial graph.
It is consequently a simple closed curve, even when it is mildly nonconvex.
Clearances use dense contour nodes with a conservative analytic arc-covering
radius, rather than bounding circles or node distances alone. Curvature is
checked at two angular resolutions; this is a sampled regularity check, not
an interval-arithmetic certificate of the continuous curvature extremum.
"""
import numpy as np
from scipy.spatial import cKDTree


def _positive_integer(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f'{label} must be a positive integer')


def _range(value, label, positive):
    a = np.asarray(value, dtype=float)
    if a.shape != (2,) or not np.isfinite(a).all() or a[0] > a[1]:
        raise ValueError(f'{label} must be a finite ordered pair')
    if positive and a[0] <= 0:
        raise ValueError(f'{label} must be strictly positive')
    return a


def _validate_config(c):
    g, p, v = c.geometry, c.particles, c.validation
    for name in ('count', 'minimum_near_pit_count', 'max_particle_attempts',
                 'max_geometry_attempts'):
        _positive_integer(p[name], f'particles.{name}')
    for name in ('boundary_points', 'refined_points', 'area_points'):
        _positive_integer(v[name], f'validation.{name}')
    if v.boundary_points < 64 or v.refined_points != 2*v.boundary_points:
        raise ValueError('Use at least 64 boundary points and exactly twice as many refined points')
    if v.area_points < 2*v.refined_points:
        raise ValueError('Area normalization needs at least twice the refined contour count')
    positive = [g.length, g.height, g.pit_radius, g.mouth_half_width, g.interface_k,
                p.wall_clearance, p.particle_clearance, p.pit_clearance,
                p.minimum_curvature_radius, p.near_pit_distance,
                v.area_relative_tolerance, v.gap_refinement_tolerance,
                v.curvature_relative_tolerance, v.membership_tolerance]
    if not np.isfinite(positive).all() or min(positive) <= 0:
        raise ValueError('Dimensions, separations and validation tolerances must be positive and finite')
    if not np.isfinite(g.pit_x) or not g.pit_radius < g.pit_x < g.length-g.pit_radius:
        raise ValueError('Initial semicircular pit must fit the upper boundary')
    if g.mouth_half_width >= g.pit_radius or g.pit_radius >= g.height:
        raise ValueError('Opening must fit the pit and the pit must fit the domain')
    rx = _range(p.center_x_range, 'center_x_range', False)
    ry = _range(p.center_y_range, 'center_y_range', False)
    _range(p.equivalent_radius_range, 'equivalent_radius_range', True)
    ar = _range(p.aspect_range, 'aspect_range', True)
    if ar[0] < 1 or rx[0] < 0 or rx[1] > g.length or ry[0] < 0 or ry[1] > g.height:
        raise ValueError('Aspect must be >=1 and center ranges must fit the rectangle')
    modes = np.asarray(p.modes)
    if (modes.ndim != 1 or not len(modes) or not np.issubdtype(modes.dtype, np.integer)
            or np.any(modes < 3) or np.any(np.diff(modes) <= 0)):
        raise ValueError('Radial modes must be strictly increasing integers >=3')
    scales, limits = np.asarray(p.coefficient_std), np.asarray(p.coefficient_max)
    if (scales.shape != modes.shape or limits.shape != modes.shape or
            not np.isfinite(scales).all() or not np.isfinite(limits).all() or
            np.any(scales <= 0) or np.any(limits <= 0)):
        raise ValueError('Each radial mode needs a positive finite coefficient std and limit')
    if p.minimum_near_pit_count > p.count or p.near_pit_distance <= p.pit_clearance:
        raise ValueError('Near-pit count/distance must permit the requested particles')


def _particle_arrays(particle):
    center = np.asarray(particle['center'], dtype=float)
    modes = np.asarray(particle['modes'])
    a, b = np.asarray(particle['cosine'], dtype=float), np.asarray(particle['sine'], dtype=float)
    scalar = np.asarray([particle['equivalent_radius'], particle['aspect'],
                         particle['angle'], particle['radial_scale']], dtype=float)
    if (center.shape != (2,) or not np.isfinite(center).all() or modes.ndim != 1 or
            not len(modes) or not np.issubdtype(modes.dtype, np.integer) or
            np.any(modes < 3) or np.any(np.diff(modes) <= 0) or a.shape != modes.shape or
            b.shape != modes.shape or not np.isfinite(a).all() or not np.isfinite(b).all() or
            not np.isfinite(scalar).all() or scalar[0] <= 0 or scalar[1] < 1 or scalar[3] <= 0):
        raise ValueError('Invalid positive radial particle representation')
    angle, aspect = scalar[2], scalar[1]
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    affine = rotation @ np.diag([np.sqrt(aspect), 1/np.sqrt(aspect)])
    return center, modes, a, b, scalar[3], affine


def _radial(theta, modes, a, b, radial_scale):
    phases = theta[..., None]*modes
    co, si = np.cos(phases), np.sin(phases)
    f = np.sum(a*co+b*si, axis=-1)
    fp = np.sum(modes*(-a*si+b*co), axis=-1)
    fpp = -np.sum(modes**2*(a*co+b*si), axis=-1)
    radius = radial_scale*np.exp(f)
    return radius, radius*fp, radius*(fp**2+fpp)


def boundary(particle, theta):
    """Return points, ceramic-outward unit normals and signed curvature.

    Arrays retain the shape of theta, with a final coordinate dimension of 2
    for points/normals. For a PDE defined outside the particle, its hole-wall
    outward normal is the NEGATIVE of the returned ceramic-outward normal.
    Positive curvature denotes a convex counterclockwise contour.
    """
    theta = np.asarray(theta, dtype=float)
    if not np.isfinite(theta).all():
        raise ValueError('Boundary angles must be finite')
    center, modes, a, b, scale, affine = _particle_arrays(particle)
    r, rp, rpp = _radial(theta, modes, a, b, scale)
    e = np.stack([np.cos(theta), np.sin(theta)], axis=-1)
    ep = np.stack([-np.sin(theta), np.cos(theta)], axis=-1)
    points = (r[..., None]*e) @ affine.T+center
    first = (rp[..., None]*e+r[..., None]*ep) @ affine.T
    second = ((rpp-r)[..., None]*e+2*rp[..., None]*ep) @ affine.T
    speed = np.linalg.norm(first, axis=-1)
    normals = np.stack([first[..., 1], -first[..., 0]], axis=-1)/speed[..., None]
    curvature = (first[..., 0]*second[..., 1]-first[..., 1]*second[..., 0])/speed**3
    return points, normals, curvature


def _covering_radius(particle, count):
    """Upper bound on distance to the closest uniform angular contour node.

    |x'(theta)| <= sqrt(aspect)*scale*exp(sum amplitude)*sqrt(1+F1^2),
    where F1=sum(k*amplitude). Each point is within pi/count radians of a
    node; integrating this speed bound gives a conservative arc covering.
    """
    _, modes, a, b, scale, _ = _particle_arrays(particle)
    amplitude = np.hypot(a, b)
    speed_bound = np.sqrt(particle['aspect'])*scale*np.exp(amplitude.sum())*np.sqrt(
        1+np.sum(modes*amplitude)**2)
    return float(speed_bound*np.pi/count)


def _contours(particles, count):
    theta = np.arange(count)*(2*np.pi/count)
    samples = [boundary(p, theta) for p in particles]
    return ([s[0] for s in samples], [_covering_radius(p, count) for p in particles],
            [float(np.max(np.abs(s[2]))) for s in samples])


def _inside_particle(points, particle, tolerance):
    center, modes, a, b, scale, affine = _particle_arrays(particle)
    local = (np.asarray(points)-center) @ np.linalg.inv(affine).T
    theta = np.arctan2(local[..., 1], local[..., 0])
    radius = _radial(theta, modes, a, b, scale)[0]
    return np.linalg.norm(local, axis=-1) < radius-tolerance


def _separations(particles, c, count):
    nodes, cover, curvature = _contours(particles, count)
    g = c.geometry
    wall, pit_lower, pit_upper, pairs = [], [], [], []
    centers_contained = False
    for x, error in zip(nodes, cover):
        wall.append(float(min(x[:, 0].min(), g.length-x[:, 0].max(),
                              x[:, 1].min(), g.height-x[:, 1].max())-error))
        distance = float(np.linalg.norm(x-np.array([g.pit_x, 0.]), axis=1).min()-g.pit_radius)
        pit_lower.append(distance-error)
        pit_upper.append(distance)
    for i, x in enumerate(nodes):
        tree = cKDTree(x)
        for j in range(i):
            nearest = float(tree.query(nodes[j], k=1)[0].min())
            pairs.append(nearest-cover[i]-cover[j])
            # Positive boundary distance alone would also permit nesting.
            centers_contained |= bool(_inside_particle(particles[i]['center'], particles[j], 0.) or
                                      _inside_particle(particles[j]['center'], particles[i], 0.))
    return dict(wall_clearance_lower_bound=min(wall),
        pair_clearance_lower_bound=min(pairs) if pairs else None,
        pit_clearance_lower_bound=min(pit_lower),
        pit_distance_upper_bounds=pit_upper,
        minimum_sampled_curvature_radius=1/max(curvature),
        maximum_arc_covering_radius=max(cover),
        contains_particle_center=centers_contained)


def _admissible(particles, c, count):
    metrics = _separations(particles, c, count)
    p = c.particles
    return (not metrics['contains_particle_center'] and
            metrics['wall_clearance_lower_bound'] >= p.wall_clearance and
            metrics['pit_clearance_lower_bound'] >= p.pit_clearance and
            (metrics['pair_clearance_lower_bound'] is None or
             metrics['pair_clearance_lower_bound'] >= p.particle_clearance) and
            metrics['minimum_sampled_curvature_radius'] >= p.minimum_curvature_radius)


def _validate_particle_prior(particle, c):
    _particle_arrays(particle)
    p = c.particles
    for key, limits in (('equivalent_radius', p.equivalent_radius_range),
                        ('aspect', p.aspect_range)):
        if not limits[0] <= particle[key] <= limits[1]:
            raise ValueError(f'Particle {key} is outside the configured prior')
    center = particle['center']
    if (not p.center_x_range[0] <= center[0] <= p.center_x_range[1] or
            not p.center_y_range[0] <= center[1] <= p.center_y_range[1]):
        raise ValueError('Particle center is outside the configured prior')
    if tuple(particle['modes']) != tuple(p.modes) or not 0 <= particle['angle'] < 2*np.pi:
        raise ValueError('Particle modes or orientation differ from the configured prior')
    if (np.any(np.abs(particle['cosine']) > np.asarray(p.coefficient_max)) or
            np.any(np.abs(particle['sine']) > np.asarray(p.coefficient_max))):
        raise ValueError('Particle coefficient exceeds the explicit truncation limit')


def validate_geometry(record, c):
    """Validate priors, conservative clearances, area and refined curvature."""
    _validate_config(c)
    particles = record['particles']
    if record['schema'] != 'smooth_radial_particles_v1' or len(particles) != c.particles.count:
        raise ValueError('Geometry schema or particle count does not match the configuration')
    for particle in particles:
        _validate_particle_prior(particle, c)
    v = c.validation
    coarse = _separations(particles, c, v.boundary_points)
    fine = _separations(particles, c, v.refined_points)
    if not _admissible(particles, c, v.boundary_points) or not _admissible(particles, c, v.refined_points):
        raise ValueError('Particle geometry violates wall, pair, pit or curvature clearance')
    gaps = ('wall_clearance_lower_bound', 'pair_clearance_lower_bound', 'pit_clearance_lower_bound')
    gap_change = max(abs(fine[k]-coarse[k]) for k in gaps if fine[k] is not None)
    if gap_change > v.gap_refinement_tolerance:
        raise ValueError('Contour-refinement gap change exceeds the declared tolerance')
    radius_change = abs(fine['minimum_sampled_curvature_radius']/coarse[
        'minimum_sampled_curvature_radius']-1)
    if radius_change > v.curvature_relative_tolerance:
        raise ValueError('Contour-refinement curvature change exceeds the declared tolerance')
    near_count = sum(distance < c.particles.near_pit_distance for distance in fine[
        'pit_distance_upper_bounds'])
    if near_count < c.particles.minimum_near_pit_count:
        raise ValueError('Too few particles satisfy the explicitly prescribed near-pit condition')
    areas = []
    theta = np.arange(v.area_points)*(2*np.pi/v.area_points)
    for particle in particles:
        _, modes, a, b, scale, _ = _particle_arrays(particle)
        radius = _radial(theta, modes, a, b, scale)[0]
        area = float(np.pi*np.mean(radius**2))  # det(affine)=1.
        expected = np.pi*particle['equivalent_radius']**2
        refined = _radial(np.arange(2*v.area_points)*(np.pi/v.area_points), modes, a, b, scale)[0]
        refined_area = float(np.pi*np.mean(refined**2))
        if max(abs(area/expected-1), abs(refined_area/area-1)) > v.area_relative_tolerance:
            raise ValueError('Particle area normalization fails independent quadrature refinement')
        areas.append(area)
    return dict(fine, boundary_points=v.boundary_points, refined_points=v.refined_points,
        near_pit_count=near_count, particle_areas=areas,
        total_particle_area=float(sum(areas)),
        excluded_area_fraction=float(sum(areas)/(c.geometry.length*c.geometry.height)),
        maximum_gap_refinement_change=gap_change,
        curvature_radius_relative_refinement_change=radius_change,
        clearance_method='nearest contour nodes minus analytic arc-covering bounds',
        curvature_method='sampled analytic curvature at two angular resolutions')


def _draw_particle(c, rng):
    p = c.particles
    coefficients = rng.normal(size=(2, len(p.modes)))*np.asarray(p.coefficient_std)
    if np.any(np.abs(coefficients) > np.asarray(p.coefficient_max)):
        return None
    radius = float(rng.uniform(*p.equivalent_radius_range))
    a, b = coefficients
    theta = np.arange(c.validation.area_points)*(2*np.pi/c.validation.area_points)
    phases = theta[:, None]*np.asarray(p.modes)
    factor = np.sqrt(np.mean(np.exp(2*np.sum(a*np.cos(phases)+b*np.sin(phases), axis=1))))
    return dict(center=[float(rng.uniform(*p.center_x_range)), float(rng.uniform(*p.center_y_range))],
        equivalent_radius=radius, aspect=float(rng.uniform(*p.aspect_range)),
        angle=float(rng.uniform(0, 2*np.pi)), modes=list(p.modes),
        cosine=a.tolist(), sine=b.tolist(), radial_scale=float(radius/factor))


class GeometrySamplingExhausted(RuntimeError):
    """A finite rejection budget ended; the caller may continue the same RNG."""
    def __init__(self, message, diagnostics):
        super().__init__(message)
        self.diagnostics = dict(diagnostics, rejections=dict(diagnostics['rejections']))

    def __reduce__(self):
        return type(self), (str(self), self.diagnostics)


def sample_geometry(c, rng, *, diagnostics=None):
    """Draw explicitly conditioned particles; never substitute a fallback.

    Normal Fourier draws are explicitly truncated by rejection. Placement,
    separation, smoothness and near-pit acceptance define the geometry prior;
    accepted particles are therefore not independent uniform centers. Optional
    diagnostics is an output dict, reset per budget; it never consumes RNG.
    """
    _validate_config(c)
    if c.particles.placement not in ('global_rejection', 'near_first'):
        raise ValueError('Explicit particles.placement must be global_rejection or near_first')
    if not isinstance(rng, np.random.Generator):
        raise ValueError('rng must be an explicit numpy.random.Generator')
    if diagnostics is not None and not isinstance(diagnostics, dict):
        raise ValueError('Optional diagnostics must be a mutable output dict')
    counts = {} if diagnostics is None else diagnostics
    counts.clear()
    counts.update(layout_attempts=0, particle_proposals=0, accepted_particle_proposals=0,
        outcome='in_progress', rejections=dict(coefficient_bounds=0, near_pit_placement=0,
        coarse_clearance_or_curvature=0, refined_clearance_or_curvature=0,
        incomplete_layout=0, layout_near_pit_count=0, layout_gap_refinement=0,
        layout_curvature_refinement=0))
    rejected = counts['rejections']
    for _ in range(c.particles.max_geometry_attempts):
        counts['layout_attempts'] += 1
        particles = []
        for index in range(c.particles.count):
            for _ in range(c.particles.max_particle_attempts):
                counts['particle_proposals'] += 1
                candidate = _draw_particle(c, rng)
                if candidate is None:
                    rejected['coefficient_bounds'] += 1
                    continue
                if (c.particles.placement == 'near_first'
                        and index < c.particles.minimum_near_pit_count):
                    nodes = boundary(candidate, np.arange(c.validation.boundary_points)
                                     * (2*np.pi/c.validation.boundary_points))[0]
                    distance = np.linalg.norm(nodes-np.array([c.geometry.pit_x, 0.]), axis=1).min()-c.geometry.pit_radius
                    if distance >= c.particles.near_pit_distance:
                        rejected['near_pit_placement'] += 1
                        continue
                if not _admissible(particles+[candidate], c, c.validation.boundary_points):
                    rejected['coarse_clearance_or_curvature'] += 1
                    continue
                if not _admissible(particles+[candidate], c, c.validation.refined_points):
                    rejected['refined_clearance_or_curvature'] += 1
                    continue
                particles.append(candidate)
                counts['accepted_particle_proposals'] += 1
                break
            else:
                break
        if len(particles) != c.particles.count:
            rejected['incomplete_layout'] += 1
            continue
        record = dict(schema='smooth_radial_particles_v1', particles=particles)
        metrics = _separations(particles, c, c.validation.refined_points)
        if sum(d < c.particles.near_pit_distance for d in metrics[
                'pit_distance_upper_bounds']) < c.particles.minimum_near_pit_count:
            rejected['layout_near_pit_count'] += 1
            continue
        # Refinement tolerances concern the completed layout's extrema. Applying
        # them to partial layouts would impose additional prior restrictions.
        coarse = _separations(particles, c, c.validation.boundary_points)
        gaps = ('wall_clearance_lower_bound', 'pair_clearance_lower_bound', 'pit_clearance_lower_bound')
        gap_change = max(abs(metrics[k]-coarse[k]) for k in gaps if metrics[k] is not None)
        radius_change = abs(metrics['minimum_sampled_curvature_radius']/coarse[
            'minimum_sampled_curvature_radius']-1)
        if (gap_change > c.validation.gap_refinement_tolerance or
                radius_change > c.validation.curvature_relative_tolerance):
            rejected['layout_gap_refinement'] += int(gap_change > c.validation.gap_refinement_tolerance)
            rejected['layout_curvature_refinement'] += int(radius_change > c.validation.curvature_relative_tolerance)
            continue
        # Both resolutions already gate candidate admission. Keep the independent
        # validator: structural, normalization and unexpected errors still raise.
        validate_geometry(record, c)
        # Particle identities are not physical inputs. Randomize ordering after
        # near-first placement; conditioning and operators are permutation-free.
        if c.particles.placement == 'near_first':
            rng.shuffle(record['particles'])
        counts['outcome'] = 'accepted'
        return record
    counts['outcome'] = 'budget_exhausted'
    raise GeometrySamplingExhausted('Unable to sample the declared particle geometry within '
        f'{c.particles.max_geometry_attempts} geometry attempts and '
        f'{c.particles.max_particle_attempts} proposals per particle', counts)


def contains(points, record, c):
    """Boolean mask for the rectangle outside ceramic interiors (wall included).

    Invert each particle's affine map, recover polar angle, and compare its
    radius to the analytic radial graph. No polygon rasterization or lookup
    grid is used. The initial electrolyte pit remains inside the PDE domain.
    """
    _validate_config(c)
    points = np.asarray(points, dtype=float)
    if points.ndim < 1 or points.shape[-1] != 2 or not np.isfinite(points).all():
        raise ValueError('Query points must be finite with a final coordinate dimension of 2')
    if record['schema'] != 'smooth_radial_particles_v1' or len(record['particles']) != c.particles.count:
        raise ValueError('Geometry schema or particle count does not match the configuration')
    tol = c.validation.membership_tolerance
    inside = ((points[..., 0] >= -tol) & (points[..., 0] <= c.geometry.length+tol) &
              (points[..., 1] >= -tol) & (points[..., 1] <= c.geometry.height+tol))
    for particle in record['particles']:
        inside &= ~_inside_particle(points, particle, tol)
    return inside
