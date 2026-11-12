"""Independent local finite-difference and native-graph sphere references.

Reflection symmetry reduces the cube to one octant; it does not impose a
spherical evolution law. At the center planes the first node is h/2 away,
so reflection repeats that node. At the physical faces, local Neumann ghost
values reflect the penultimate node. The native graph instead omits missing
edges, exactly as the original regional graph. Only evaluation arrays are saved.
"""
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np
from scipy.interpolate import RegularGridInterpolator


def parameters(c):
    p, nd = c.physical, c.nondim
    a = p.M*nd.T_c/nd.L_c**2
    return a, a/p.eps_star**2, p.lam*nd.T_c/(nd.L_c*np.sqrt(2)*p.eps_star)


def graph_weight(c):
    h = 1/(c.sample.Nx-1)
    if not h < c.nl.delta < np.sqrt(2)*h:
        raise ValueError('Thickness control requires the declared six-axis cubic horizon')
    return 315/(2*np.pi*c.nl.delta**5)*(1-h/c.nl.delta)**2*h**3


def octant_laplacian(phi, h, *, local):
    result = jnp.zeros_like(phi)
    for axis in range(3):
        first = jnp.take(phi, jnp.array([0]), axis=axis)
        last = jnp.take(phi, jnp.array([-2 if local else -1]), axis=axis)
        padded = jnp.concatenate((first, phi, last), axis=axis)
        n = phi.shape[axis]
        result = result + (jax.lax.slice_in_dim(padded, 0, n, axis=axis)
            + jax.lax.slice_in_dim(padded, 2, n+2, axis=axis)-2*phi)
    return result/h**2


def latent_diffusion(u, h, *, local):
    """Local chain rule or exact graph difference ratio; no inverse-tanh rounding.

    Delta(phi)/sech(u)^2 = Delta(u)-2*tanh(u)*|grad(u)|^2.
    The nonlocal branch instead retains every exact finite difference of phi,
    divided by the center Jacobian, using a stable hyperbolic identity.
    """
    if local:
        gradients = []
        for axis in range(3):
            first = jnp.take(u, jnp.array([0]), axis=axis)
            last = jnp.take(u, jnp.array([-2]), axis=axis)
            padded = jnp.concatenate((first, u, last), axis=axis)
            n = u.shape[axis]
            gradients.append((jax.lax.slice_in_dim(padded, 2, n+2, axis=axis)
                - jax.lax.slice_in_dim(padded, 0, n, axis=axis))/(2*h))
        return octant_laplacian(u, h, local=True)-2*jnp.tanh(u)*sum(g*g for g in gradients)
    total = jnp.zeros_like(u)
    for axis in range(3):
        padded = jnp.concatenate((jnp.take(u, jnp.array([0]), axis=axis), u,
            jnp.take(u, jnp.array([-1]), axis=axis)), axis=axis)
        for start in (0, 2):
            v = jax.lax.slice_in_dim(padded, start, start+u.shape[axis], axis=axis)
            total = total+jnp.sinh(v-u)*jnp.exp(jnp.logaddexp(u, -u)-jnp.logaddexp(v, -v))
    return total/h**2


def integrate(c, n, dt, *, local):
    if not jax.config.x64_enabled or n % 2 or n < 4:
        raise ValueError('Reference requires FP64 and an even full-cube grid')
    stride = round(1/((c.experiment.snapshots-1)*dt))
    if stride < 1 or not np.isclose(stride*dt, 1/(c.experiment.snapshots-1), atol=1e-14):
        raise ValueError('Reference dt must divide saved time intervals')
    h = 1/(n-1)
    a, b, g = parameters(c)
    if not local:
        if n != c.sample.Nx:
            raise ValueError('Native reference uses the training graph resolution')
        a *= graph_weight(c)*h*h
    if dt*(12*a/h**2+2*b+2*g) >= 2:
        raise ValueError('Explicit reference step exceeds the conservative stability screen')
    axis = (np.arange(n//2)+.5)*h
    xyz = np.meshgrid(axis, axis, axis, indexing='ij', sparse=True)
    initial = (c.physical.R0/c.nondim.L_c
        - np.sqrt(sum(v*v for v in xyz)))/(np.sqrt(2)*c.physical.eps_star)
    form = c.experiment.reference_form
    if form not in ('phase', 'latent'):
        raise ValueError('Declare reference_form=phase or latent explicitly')
    if form == 'phase':
        initial = np.tanh(initial)

    def rhs(phi):
        if form == 'latent':
            return a*latent_diffusion(phi, h, local=local)+b*jnp.tanh(phi)-g
        return a*octant_laplacian(phi, h, local=local)-b*(phi**3-phi)-g*(1-phi**2)

    @jax.jit
    def advance(phi):
        def step(phi, _):
            k1 = rhs(phi)
            k2 = rhs(phi+dt*k1/2)
            k3 = rhs(phi+dt*k2/2)
            k4 = rhs(phi+dt*k3)
            return phi+dt*(k1+2*k2+2*k3+k4)/6, None
        return jax.lax.scan(step, phi, None, length=stride)[0]

    phi = jnp.asarray(initial, dtype=jnp.float64)
    for k in range(c.experiment.snapshots):
        if not np.isfinite(np.asarray(phi)).all():
            raise FloatingPointError(f'Nonfinite {form} reference state at snapshot{k}')
        actual = np.asarray(jnp.tanh(phi) if form == 'latent' else phi)
        if not np.isfinite(actual).all() or np.max(np.abs(actual)) > 1.000001:
            raise FloatingPointError(f'Invalid reference field at snapshot{k}')
        yield axis, actual
        if k+1 < c.experiment.snapshots:
            phi = advance(phi)


def evaluation_points(c):
    axis = np.linspace(0, 1, c.experiment.evaluation_axis)
    points = np.stack(np.meshgrid(axis, axis, axis, indexing='ij'), -1).reshape(-1, 3)
    nr, nd = c.experiment.radial_points, c.experiment.radial_directions
    z = 1-2*(np.arange(nd)+.5)/nd
    angle = np.arange(nd)*np.pi*(3-np.sqrt(5))
    directions = np.stack((np.sqrt(1-z*z)*np.cos(angle), np.sqrt(1-z*z)*np.sin(angle), z), -1)
    radius = np.linspace(.002, .49, nr)
    rays = .5+directions[:, None, :]*radius[None, :, None]
    return points, rays, radius


def sample_reference(c, n, dt, *, local):
    points, rays, radius = evaluation_points(c)
    query = np.concatenate((points, rays.reshape(-1, 3)))
    folded = np.abs(query-.5)
    sampled = []
    start = time.perf_counter()
    for k, (axis, field) in enumerate(integrate(c, n, dt, local=local)):
        # Reflection gives an even continuation up to the first h/2 node.
        coords = np.maximum(folded, axis[0])
        values = RegularGridInterpolator((axis,)*3, field, bounds_error=True)(coords)
        if k == 0:
            values = np.tanh((c.physical.R0/c.nondim.L_c-np.linalg.norm(query-.5, axis=1))
                             /(np.sqrt(2)*c.physical.eps_star))
        sampled.append(values)
    return np.stack(sampled), time.perf_counter()-start


def prepare(c, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    points, rays, radius = evaluation_points(c)
    grids, dt = c.experiment.reference_grids, c.experiment.reference_dt
    if len(grids) != 2 or grids[0] >= grids[1]:
        raise ValueError('Declare an increasing coarse/fine reference grid pair')
    coarse, t0 = sample_reference(c, grids[0], dt, local=True)
    fine, t1 = sample_reference(c, grids[1], dt, local=True)
    native, t2 = sample_reference(c, c.sample.Nx, dt, local=False)
    band = np.abs(fine[:, :len(points)]) < .9
    diff = fine[:, :len(points)]-coarse[:, :len(points)]
    report = dict(local_space_relative_l2=float(np.linalg.norm(diff)/np.linalg.norm(fine[:, :len(points)])),
        local_space_interface_relative_l2=float(np.linalg.norm(diff[band])/np.linalg.norm(fine[:, :len(points)][band])),
        seconds=[t0, t1, t2], grids=list(grids), dt=dt,
        native_second_moment=float(graph_weight(c)/(c.sample.Nx-1)**2),
        representation=c.experiment.reference_form,
        reference='Independent local 3D finite-difference RK4 with Neumann faces; cube octant symmetry only')
    (output/'config.json').write_text(c.to_json_best_effort()+'\n')
    (output/'validation.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    if report['local_space_relative_l2'] > .003 or report['local_space_interface_relative_l2'] > .03:
        raise ValueError(f'Local spatial reference screen failed: {report}')
    np.savez_compressed(output/'reference.npz', points=points, rays=rays, radius=radius,
        time=np.linspace(0, 1, c.experiment.snapshots),
        local_coarse=coarse[:, :len(points)],
        local=fine[:, :len(points)], native=native[:, :len(points)],
        local_rays=fine[:, len(points):].reshape(c.experiment.snapshots, *rays.shape[:2]),
        native_rays=native[:, len(points):].reshape(c.experiment.snapshots, *rays.shape[:2]))
    print('REFERENCE', json.dumps(report), flush=True)
