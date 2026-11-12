"""Explicit compact cubic operators for the interface/horizon research study.

The operator represents ``D(phi)_i = sum_j w_ij * (phi_i - phi_j)``.
Uniform Cartesian volumes are h**3; neighbors outside the cube are omitted.
``normalization=moment`` divides every edge by the same interior second moment,
never by a boundary row's truncated moment. This preserves symmetric positive
pair weights while making interior quadratic derivatives exact. Production
operator defaults and the earlier six-neighbor study remain unchanged.
"""
from numbers import Real

import numpy as np


def _settings(c):
    counts = (c.sample.Nx, c.sample.Ny, c.sample.Nz)
    if any(type(n) is not int or n < 4 for n in counts) or len(set(counts)) != 1:
        raise ValueError('Compact interface operator requires a uniform cubic grid with integer Nx=Ny=Nz>=4')
    delta = c.nl.delta
    if isinstance(delta, bool) or not isinstance(delta, Real) or not np.isfinite(delta):
        raise ValueError('The cubic horizon delta must be a finite real scalar')
    n, h = counts[0], 1/(counts[0]-1)
    if not h < delta < .5:
        raise ValueError(f'The local-horizon study requires h < delta < 0.5; h={h}, delta={delta}')
    if c.nl.kernel != 'cubic':
        raise ValueError('The compact interface study explicitly supports only the cubic kernel')
    if c.nl.normalization not in ('raw', 'moment'):
        raise ValueError('Declare nl.normalization as raw or moment')
    return n, h, float(delta), c.nl.normalization


def offsets_and_weights(c):
    """Return integer offsets, positive FP64 weights, and JSON-safe diagnostics.

    The raw kernel is ``315/(2*pi*delta**5) * (1-r/delta)**2 * h**3``.
    For the full symmetric neighborhood, ``M = .5 * sum(w * xi outer xi)``
    is isotropic. Taylor expansion gives ``D(phi) = -M:Hessian(phi)+...``.
    Dividing by ``trace(M)/3`` therefore enforces the local Laplacian's
    quadratic coefficient without changing any physical PDE coefficient.
    """
    n, h, delta, normalization = _settings(c)
    limit = int(np.ceil(delta/h))
    axis = np.arange(-limit, limit+1, dtype=np.int32)
    candidates = np.stack(np.meshgrid(axis, axis, axis, indexing='ij'), -1).reshape(-1, 3)
    distance = h*np.linalg.norm(candidates, axis=1)
    candidates = candidates[(distance > 0) & (distance < delta)]
    # Preserve the exact historical six-neighbor order; additional shells follow.
    axial = np.concatenate((np.eye(3, dtype=np.int32), -np.eye(3, dtype=np.int32)))
    others = candidates[np.sum(candidates*candidates, axis=1) != 1]
    offsets = np.concatenate((axial, others)).astype(np.int32)
    displacement = h*offsets
    distance = np.linalg.norm(displacement, axis=1)
    raw = 315/(2*np.pi*delta**5)*(1-distance/delta)**2*h**3
    raw_moment = .5*np.einsum('i,ij,ik->jk', raw, displacement, displacement)
    raw_m2 = float(np.trace(raw_moment)/3)
    if (not np.isfinite(raw).all() or not np.all(raw > 0)
            or not np.isfinite(raw_m2) or raw_m2 <= 0):
        raise ValueError('Cubic stencil weights and second moment must be finite and positive')
    if not np.allclose(raw_moment, raw_m2*np.eye(3), rtol=1e-12, atol=1e-13*raw_m2):
        raise ValueError('The declared full Cartesian neighborhood must have an isotropic second moment')
    weights = raw if normalization == 'raw' else raw/raw_m2
    moment = .5*np.einsum('i,ij,ik->jk', weights, displacement, displacement)
    if not np.isfinite(weights).all() or not np.all(weights > 0):
        raise ValueError('Normalized stencil weights must be finite and positive')
    metadata = dict(grid_nodes=n, h=h, delta=delta, delta_over_h=delta/h,
        kernel='cubic', normalization=normalization, neighbors=len(offsets),
        m2_raw=raw_m2, m2_effective=float(np.trace(moment)/3),
        second_moment_matrix=moment.tolist(), first_moment=(weights@displacement).tolist(),
        maximum_neighbor_distance=float(distance.max()),
        boundary='omit outside-cube edges; no boundary-row renormalization')
    return offsets, weights.astype(np.float64), metadata


def compact_stencil(ids, c, operator):
    """Gather fixed-size core-plus-halo rows from a precomputed operator.

    Every core owns all halo slots, including repeated physical nodes. An
    outside-domain slot repeats its core coordinate and carries zero weight.
    Only the first K rows have PDE weights; this matches the latent residual's
    existing ``(row, col, negative_offdiag, positive_diag)`` protocol.
    """
    n, h, delta, normalization = _settings(c)
    offsets, weights, metadata = operator
    if (metadata['grid_nodes'] != n or metadata['h'] != h or metadata['delta'] != delta
            or metadata['kernel'] != c.nl.kernel or metadata['normalization'] != normalization):
        raise ValueError('Precomputed compact operator does not match the declared grid/horizon/normalization')
    ids = np.asarray(ids)
    if (ids.ndim != 2 or ids.shape[1] != 3 or len(ids) < 1
            or not np.issubdtype(ids.dtype, np.integer)
            or np.any(ids < 0) or np.any(ids >= n)):
        raise ValueError('Core IDs must be a nonempty integer (K,3) array inside the declared grid')
    if (offsets.ndim != 2 or offsets.shape[1] != 3
            or not np.issubdtype(offsets.dtype, np.integer)
            or weights.shape != (len(offsets),) or metadata['neighbors'] != len(offsets)
            or not np.isfinite(weights).all() or np.any(weights <= 0)):
        raise ValueError('Malformed precomputed positive-weight stencil')
    k, neighbors = len(ids), len(offsets)
    if k*(neighbors+1) >= np.iinfo(np.int32).max:
        raise ValueError('Compact node count exceeds the int32 indexing capacity')
    adjacent = ids[:, None, :]+offsets[None]
    valid = np.all((adjacent >= 0) & (adjacent < n), axis=-1)
    adjacent = np.where(valid[..., None], adjacent, ids[:, None, :])
    points = (np.concatenate((ids, adjacent.reshape(-1, 3)))/(n-1)).astype(np.float32)
    row = np.repeat(np.arange(k, dtype=np.int32), neighbors)
    col = np.arange(k, (neighbors+1)*k, dtype=np.int32)
    # Sum the actual FP32 training edges, preserving the legacy row arithmetic.
    values = (-weights[None]*valid).reshape(-1).astype(np.float32)
    diagonal = np.pad(-values.reshape(k, neighbors).sum(axis=1), (0, neighbors*k))
    diagonal = diagonal.astype(np.float32)
    if not np.isfinite(values).all() or not np.isfinite(diagonal).all():
        raise ValueError('Compact operator coefficients overflow the declared FP32 training dtype')
    return points, (row, col, values, diagonal)
