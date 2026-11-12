"""Independent radial reference for the extreme-interface melting study.

The local three-dimensional PDE is
phi_t = a*Delta(phi) - b*(phi**3-phi) - g*(1-phi**2).
For a radial field phi=tanh(u), the chain rule gives
u_t = a*(u_rr + 2*u_r/r - 2*tanh(u)*u_r**2) + b*tanh(u) - g.
This solver evolves that PDE; it never prescribes an interface trajectory.

The spherical outer boundary is a far-field approximation to the cube, not
an exact symmetry reduction of its boundary condition. Cross-checks against
the independent cube solver and an outer-radius sensitivity test qualify its
use. This module imports no neural network, JAX or nonlocal operator code.
"""
from dataclasses import dataclass
import json
from pathlib import Path
import time

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import PchipInterpolator
from scipy.sparse import diags


SPACE_L2_TOLERANCE = 1e-5
SPACE_INTERFACE_TOLERANCE = 1e-3
TIME_L2_TOLERANCE = 1e-6
TIME_INTERFACE_TOLERANCE = 1e-4
OUTER_L2_TOLERANCE = 1e-6
OUTER_INTERFACE_TOLERANCE = 1e-4
MAX_PHASE_DIFFERENCE = 1e-4


@dataclass(frozen=True)
class RadialSolution:
    radius: np.ndarray
    time: np.ndarray
    latent: np.ndarray
    statistics: dict


def _positive(value, name):
    if isinstance(value, bool) or not np.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be finite and positive')


def _integer(value, name, minimum):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')


def parameters(c):
    """Derive coefficients independently from dimensional material/scales."""
    p, nd = c.physical, c.nondim
    for name, value in [('M', p.M), ('lambda', p.lam), ('R0', p.R0),
                        ('epsilon', p.eps_star), ('L_c', nd.L_c), ('T_c', nd.T_c)]:
        _positive(value, name)
    if tuple(nd.xc) != (.5, .5, .5) or not 0 < p.R0/nd.L_c < .5:
        raise ValueError('Radial study requires a centered sphere inside the unit cube')
    a = p.M*nd.T_c/nd.L_c**2
    return a, a/p.eps_star**2, p.lam*nd.T_c/(nd.L_c*np.sqrt(2)*p.eps_star)


def radial_system(radius, a, b, g):
    """Centered radial differences and their analytic tridiagonal Jacobian.

    Even reflection gives Delta(u)(0)=3*u_rr(0)=6*(u[1]-u[0])/h**2.
    The outer ghost implements zero radial derivative. Both endpoint gradient
    squares are zero, including in the analytic Jacobian.
    """
    radius = np.asarray(radius, dtype=np.float64)
    if (radius.ndim != 1 or len(radius) < 3 or radius[0] != 0
            or not np.isfinite(radius).all() or not np.all(np.diff(radius) > 0)):
        raise ValueError('Require a finite increasing radial grid starting at zero')
    h = radius[1]
    if not np.allclose(np.diff(radius), h, rtol=1e-11, atol=1e-14):
        raise ValueError('Radial differences require a uniform grid')
    for name, value in [('a', a), ('b', b), ('g', g)]:
        _positive(value, name)
    count = len(radius)
    lower = np.zeros(count)
    upper = np.zeros(count)
    diagonal = np.full(count, -2*a/h**2)
    lower[1:-1] = a/h**2-a/(h*radius[1:-1])
    upper[1:-1] = a/h**2+a/(h*radius[1:-1])
    diagonal[0], upper[0], lower[-1] = -6*a/h**2, 6*a/h**2, 2*a/h**2
    laplacian = diags((lower[1:], diagonal, upper[:-1]), (-1, 0, 1), format='csc')

    def rhs(_time, u):
        gradient = np.zeros_like(u)
        gradient[1:-1] = (u[2:]-u[:-2])/(2*h)
        phase = np.tanh(u)
        return laplacian@u-2*a*phase*gradient**2+b*phase-g

    def jacobian(_time, u):
        gradient = np.zeros_like(u)
        gradient[1:-1] = (u[2:]-u[:-2])/(2*h)
        phase = np.tanh(u)
        lo, hi = lower.copy(), upper.copy()
        lo[1:-1] += 2*a*phase[1:-1]*gradient[1:-1]/h
        hi[1:-1] -= 2*a*phase[1:-1]*gradient[1:-1]/h
        center = diagonal+(-2*a*gradient**2+b)*(1-phase**2)
        return diags((lo[1:], center, hi[:-1]), (-1, 0, 1), format='csc')

    return rhs, jacobian


def solve_radial(c, *, nodes, outer_radius, max_step, rtol, atol):
    """Integrate the independent FP64 radial latent PDE with SciPy BDF."""
    _integer(nodes, 'radial nodes', 3)
    _integer(c.experiment.snapshots, 'snapshots', 2)
    for name, value in [('outer radius', outer_radius), ('maximum time step', max_step),
                        ('rtol', rtol), ('atol', atol)]:
        _positive(value, name)
    if outer_radius <= np.sqrt(3)/2:
        raise ValueError('The radial domain must cover every point of the unit cube')
    if max_step > 1 or rtol >= 1 or atol >= 1:
        raise ValueError('Reference tolerances and time step must resolve the unit time interval')
    radius = np.linspace(0., outer_radius, nodes)
    a, b, g = parameters(c)
    rhs, jacobian = radial_system(radius, a, b, g)
    initial = (c.physical.R0/c.nondim.L_c-radius)/(np.sqrt(2)*c.physical.eps_star)
    times = np.linspace(0., 1., c.experiment.snapshots)
    start = time.perf_counter()
    result = solve_ivp(rhs, (0., 1.), initial, method='BDF', jac=jacobian,
        t_eval=times, max_step=max_step, rtol=rtol, atol=atol)
    seconds = time.perf_counter()-start
    if not result.success:
        raise RuntimeError(f'Radial BDF failed: {result.message}')
    if result.y.shape != (nodes, len(times)) or not np.isfinite(result.y).all():
        raise FloatingPointError('Radial BDF must return finite latent states at every snapshot')
    np.testing.assert_array_equal(result.t, times)
    return RadialSolution(radius, times, result.y.T,
        dict(seconds=seconds, nodes=nodes, outer_radius=outer_radius,
             radial_spacing=outer_radius/(nodes-1), max_step=max_step,
             rtol=rtol, atol=atol, nfev=result.nfev, njev=result.njev, nlu=result.nlu))


def fibonacci_directions(count):
    _integer(count, 'angular directions', 4)
    z = 1-2*(np.arange(count)+.5)/count
    angle = np.arange(count)*np.pi*(3-np.sqrt(5))
    return np.stack((np.sqrt(1-z*z)*np.cos(angle),
                     np.sqrt(1-z*z)*np.sin(angle), z), axis=-1)


def evaluation_points(c):
    """Geometry-only cubature: angular rays stop at the actual cube boundary.

    The volume element is r**2 dr dOmega. No reference interface location or
    future physical field determines any query or quadrature weight.
    """
    parameters(c)
    s = c.experiment
    _integer(s.quadrature_radial, 'quadrature radial count', 2)
    _integer(s.radial_points, 'diagnostic radial count', 2)
    directions = fibonacci_directions(s.quadrature_directions)
    end = .5/np.max(np.abs(directions), axis=1)
    dr = end/s.quadrature_radial
    radial = dr[:, None]*(np.arange(s.quadrature_radial)+.5)
    points = .5+directions[:, None, :]*radial[:, :, None]
    weights = (4*np.pi/s.quadrature_directions)*radial**2*dr[:, None]
    radius = np.linspace(.002, .49, s.radial_points)
    rays = .5+fibonacci_directions(s.radial_directions)[:, None, :]*radius[None, :, None]
    if not np.all((points > 0) & (points < 1)) or not np.all(weights > 0):
        raise ValueError('Cube cubature must have interior queries and positive volume weights')
    return points.reshape(-1, 3), weights.ravel(), rays, radius


def sample_solution(solution, points):
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError('Reference queries must be a finite N-by-3 array')
    radii = np.linalg.norm(points-.5, axis=1)
    if np.any(radii > solution.radius[-1]):
        raise ValueError('Reference query lies outside the radial computational domain')
    return np.stack([np.tanh(PchipInterpolator(solution.radius, state, extrapolate=False)(radii))
                     for state in solution.latent])


def weighted_metrics(prediction, truth, weights):
    prediction, truth, weights = map(np.asarray, (prediction, truth, weights))
    if (prediction.shape != truth.shape or truth.ndim != 2 or truth.shape[0] < 2
            or weights.shape != (truth.shape[1],) or not np.all(weights > 0)
            or not all(np.isfinite(x).all() for x in (prediction, truth, weights))):
        raise ValueError('Metrics require matching finite snapshots and positive spatial weights')
    p, y = prediction[1:], truth[1:]
    band = np.abs(y) < .9
    if not band.any():
        raise ValueError('Evaluation quadrature has no positive-time interface samples')
    def relative(delta, target, measure):
        denominator = np.sum(measure*target**2)
        if denominator <= 0:
            raise ValueError('Relative metric denominator is zero')
        return float(np.sqrt(np.sum(measure*delta**2)/denominator))
    return dict(pooled=relative(p-y, y, weights),
        terminal=relative(p[-1]-y[-1], y[-1], weights),
        interface=relative(p-y, y, weights*band),
        max_absolute=float(np.max(np.abs(p-y))))


def _check_difference(metrics, *, l2, interface, name):
    if (metrics['pooled'] > l2 or metrics['interface'] > interface
            or metrics['max_absolute'] > MAX_PHASE_DIFFERENCE):
        raise ValueError(f'{name} radial reference refinement failed: {metrics}')


def prepare(c, output):
    """Save only independently validated local fields and their provenance."""
    spec = c.reference
    grids = tuple(spec.grids)
    if len(grids) != 2 or grids[0] >= grids[1]:
        raise ValueError('Declare two increasing radial reference grids')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    # Store the complete declared config for replay; reference consumers compare
    # physical/nondimensional/reference/evaluation fields, never nl/sample data.
    (output/'config.json').write_text(c.to_json_best_effort()+'\n')
    points, weights, rays, radius = evaluation_points(c)
    queries = np.concatenate((points, rays.reshape(-1, 3)))
    def solve(nodes, outer_radius, max_step, rtol, atol):
        return solve_radial(c, nodes=nodes, outer_radius=outer_radius,
                            max_step=max_step, rtol=rtol, atol=atol)
    coarse = solve(grids[0], spec.outer_radius, spec.max_step, spec.rtol, spec.atol)
    fine = solve(grids[1], spec.outer_radius, spec.max_step, spec.rtol, spec.atol)
    coarse_values = sample_solution(coarse, points)
    fine_values = sample_solution(fine, queries)
    local = fine_values[:, :len(points)]
    space = weighted_metrics(coarse_values, local, weights)
    half = solve(grids[1], spec.outer_radius, spec.max_step/2, spec.rtol/5, spec.atol/5)
    time_refinement = weighted_metrics(sample_solution(half, points), local, weights)
    outer_radius = 1.1
    if spec.outer_radius >= outer_radius:
        raise ValueError('Declare outer_radius below the explicit1.1sensitivity domain')
    outer_nodes = round(outer_radius/(spec.outer_radius/(grids[1]-1)))+1
    outer = solve(outer_nodes, outer_radius, spec.max_step, spec.rtol, spec.atol)
    outer_difference = weighted_metrics(sample_solution(outer, points), local, weights)
    report = dict(status='checking', reference_kind='radial_latent_far_field_surrogate',
        equation='u_t=a*(u_rr+2*u_r/r-2*tanh(u)*u_r**2)+b*tanh(u)-g',
        initial='u(r,0)=(R0/Lc-r)/(sqrt(2)*epsilon)',
        boundary='u_r(0)=u_r(outer_radius)=0; spherical outer domain, not exact cube BC',
        cube_qualification='Requires an independently recorded cube/radial comparison; outer-radius sensitivity alone does not prove cube equivalence',
        physical_coefficients=list(parameters(c)),
        space=space, time=time_refinement, outer=outer_difference,
        solvers=dict(coarse=coarse.statistics, fine=fine.statistics,
                     time=half.statistics, outer=outer.statistics),
        gates=dict(space_l2=SPACE_L2_TOLERANCE, space_interface=SPACE_INTERFACE_TOLERANCE,
                   time_l2=TIME_L2_TOLERANCE, time_interface=TIME_INTERFACE_TOLERANCE,
                   outer_l2=OUTER_L2_TOLERANCE, outer_interface=OUTER_INTERFACE_TOLERANCE,
                   max_absolute=MAX_PHASE_DIFFERENCE),
        quadrature=dict(kind='fibonacci_cube_radial_midpoint',
            directions=c.experiment.quadrature_directions,
            radial=c.experiment.quadrature_radial, volume=float(weights.sum()),
            volume_error=float(weights.sum()-1), evolved_field_dependent=False,
            weights_renormalized=False),
        positive_time_metrics=True, native_reference_included=False)
    report_path = output/'validation.json'
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    _check_difference(space, l2=SPACE_L2_TOLERANCE,
                      interface=SPACE_INTERFACE_TOLERANCE, name='spatial')
    _check_difference(time_refinement, l2=TIME_L2_TOLERANCE,
                      interface=TIME_INTERFACE_TOLERANCE, name='temporal')
    _check_difference(outer_difference, l2=OUTER_L2_TOLERANCE,
                      interface=OUTER_INTERFACE_TOLERANCE, name='outer-domain')
    np.savez_compressed(output/'reference.npz', points=points, weights=weights,
        time=fine.time, local_coarse=coarse_values, local=local, rays=rays, radius=radius,
        local_rays=fine_values[:, len(points):].reshape(len(fine.time), *rays.shape[:2]))
    np.savez_compressed(output/'radial-profile.npz', radius=fine.radius,
                        time=fine.time, latent=fine.latent)
    report['status'] = 'passed'
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print('EXTREME_REFERENCE', json.dumps(report, allow_nan=False), flush=True)
