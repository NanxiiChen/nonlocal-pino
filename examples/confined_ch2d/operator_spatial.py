"""Exact residual-row sampling; the dense metric still requires all nodes.

This diagnostic is NOT a compact neural-query/one-hop-halo approximation.
Global mass projection and every dense metric column remain unchanged.
"""
import numpy as np
from nonlocal_pinn.training.spatial_sampling import stratified_indices


def residual_rows(data, count, rng):
    """Attach fixed-count, freshly sampled volume-weighted residual row IDs."""
    volumes = np.asarray(data['volumes'])
    points = np.asarray(data['points'])
    if (volumes.ndim != 1 or points.shape != (len(volumes), 2)
            or not np.issubdtype(volumes.dtype, np.floating)
            or not np.isfinite(volumes).all() or np.any(volumes < 0)
            or not np.isfinite(points).all()):
        raise ValueError('Residual sampling requires finite points and nonnegative volumes')
    ids = np.flatnonzero(volumes > 0)
    ids = ids[np.lexsort((points[ids, 1], points[ids, 0]))]
    rows = ids[stratified_indices(volumes[ids], count, rng)]
    return dict(data, residual_core_indices=rows.astype(np.int32),
                residual_core_weights=np.full(count, 1/count, dtype=volumes.dtype))
