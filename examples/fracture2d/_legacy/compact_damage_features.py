"""Continuous compact radial-hat hidden units with trainable neural readouts.

phi_j(x)=max(1-||x-center_j||/radius_j,0). Geometry and neural residual alone
choose centers; no nodal damage solve or direct training target is involved.
Fixed-H column normalization is a parameter reparameterization of the ansatz.
"""
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.spatial import cKDTree


class Readout(eqx.Module):
    coefficients: jax.Array


def residual_power_loss(rho, H, *, power, floor):
    """Degree-two smooth peak objective; physical readiness uses full residual."""
    if type(power) is not int or power < 2 or power % 2 or not 0 < floor < 1:
        raise ValueError('residual power/floor must be explicit even/positive')
    normalized = rho/(2*jnp.sqrt(jnp.mean(H**2)))
    scale = jax.lax.stop_gradient(jnp.maximum(jnp.max(abs(normalized)), floor))
    return scale**2*(jnp.mean((normalized/scale)**power)+(floor/scale)**power)**(2/power)


def build_basis(points, residual, c, *, inherited=None):
    if (points.dtype != np.float64 or points.ndim != 2 or points.shape[1] != 2
            or residual.shape != (len(points),) or not np.isfinite(points).all()
            or not np.isfinite(residual).all() or type(c.adaptive_centers) is not int
            or c.adaptive_centers < 0 or not 0 < c.grid_step < c.coarse_radius
            or not 0 < c.minimum_separation < c.minimum_radius <= c.maximum_radius
            or not np.isfinite(c.radius_factor) or c.radius_factor <= 0):
        raise ValueError('invalid explicit compact neural feature controls')
    tree = cKDTree(points)
    if inherited is None:
        axis = np.arange(0, 1+c.grid_step/2, c.grid_step)
        x, y = np.meshgrid(axis, axis)
        _, indices = tree.query(np.column_stack((x.ravel(), y.ravel())))
        initial = np.unique(indices)
        initial_radii = np.full(len(initial), c.coarse_radius)
    else:
        initial, initial_radii = inherited
        if (initial.ndim != 1 or initial.dtype != np.int64
                or initial_radii.shape != initial.shape
                or np.min(initial) < 0 or np.max(initial) >= len(points)
                or not np.isfinite(initial_radii).all() or np.min(initial_radii) <= 0):
            raise ValueError('invalid inherited continuous feature geometry')
    adaptive_radii = np.clip(c.radius_factor*tree.query(points, k=2)[0][:, 1],
                             c.minimum_radius, c.maximum_radius)
    # Select locally separated large neural residuals; center identities are
    # frozen before an optimizer starts, preserving fixed JIT array shapes.
    blocked = np.zeros(len(points), bool)
    # A coarse unit at the crack tip/corner must not exclude a finer unit at
    # the same position. Separation applies within the adaptive scale only.
    # Exact duplicate radius/center columns are excluded in a nested basis.
    for i, radius in zip(initial, initial_radii):
        if np.isclose(radius, adaptive_radii[i], rtol=1e-12, atol=0):
            blocked[i] = True
    chosen = []
    for i in (np.argsort(-abs(residual), kind='stable') if c.adaptive_centers else []):
        if not blocked[i]:
            chosen.append(i)
            blocked[tree.query_ball_point(points[i], c.minimum_separation)] = True
            if len(chosen) == c.adaptive_centers:
                break
    if len(chosen) != c.adaptive_centers:
        raise ValueError('insufficient separated adaptive neural feature centers')
    selected = np.concatenate((initial, np.asarray(chosen, np.int64)))
    centers = points[selected]
    radii = np.concatenate((initial_radii, adaptive_radii[chosen]))
    rows, columns, values = [], [], []
    for j, (center, radius) in enumerate(zip(centers, radii)):
        neighbor = np.asarray(tree.query_ball_point(center, radius), np.int64)
        phi = np.maximum(1-np.linalg.norm(points[neighbor]-center, axis=1)/radius, 0.)
        rows.extend(neighbor.tolist())
        columns.extend([j]*len(neighbor))
        values.extend(phi.tolist())
    matrix = coo_matrix((values, (rows, columns)), shape=(len(points), len(centers))).tocsr()
    matrix.eliminate_zeros()
    if np.any(np.diff(matrix.indptr) == 0):
        raise ValueError('compact basis leaves particle rows uncovered')
    return matrix, centers, radii, selected


def pack_basis(matrix, column_scale):
    if (matrix.dtype != np.float64 or column_scale.shape != (matrix.shape[1],)
            or not np.isfinite(column_scale).all() or np.min(column_scale) <= 0):
        raise ValueError('compact basis requires finite positive column scales')
    width = int(np.max(np.diff(matrix.indptr)))
    columns = np.zeros((matrix.shape[0], width), np.int32)
    weights = np.zeros_like(columns, np.float64)
    for i in range(matrix.shape[0]):
        start, end = matrix.indptr[i:i+2]
        columns[i, :end-start] = matrix.indices[start:end]
        weights[i, :end-start] = matrix.data[start:end] * column_scale[matrix.indices[start:end]]
    columns, weights = jnp.asarray(columns), jnp.asarray(weights)

    @jax.jit
    def field(net):
        return jnp.sum(weights * net.coefficients[columns], axis=1)
    return field, width


def residual_column_scale(D, matrix, H, *, gc0, gc1):
    if H.shape != (matrix.shape[0],) or np.min(H) < 0 or not np.isfinite(H).all():
        raise ValueError('invalid known history for compact basis normalization')
    action = diags(gc0+2*H) @ matrix + gc1*(D @ matrix)
    norm = np.sqrt(np.asarray(action.power(2).sum(axis=0)).ravel())
    if not np.isfinite(norm).all() or np.min(norm) <= 0:
        raise ValueError('compact neural basis has an invalid/zero AT2 column')
    return 2*np.sqrt(np.mean(H**2))*np.sqrt(len(H))/norm
