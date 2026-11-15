"""Exact CPU boundary candidate filtering for the rolling spatial pool."""
import numpy as np
from scipy.spatial import cKDTree


def indexed_band(points, segments, width):
    """Filter segment candidates, then apply the original distance predicate.

    A point within width of a segment is within width+half_length of its
    midpoint. Only this broad-phase radius is inflated for roundoff; final
    membership retains the original squared-distance comparison and dtype.
    """
    points, segments = np.asarray(points), np.asarray(segments)
    if (points.ndim != 2 or points.shape[1] != 2 or segments.ndim != 3
            or segments.shape[1:] != (2, 2) or not len(segments)
            or points.dtype != segments.dtype or points.dtype not in (np.float32, np.float64)
            or not np.isfinite(points).all() or not np.isfinite(segments).all()
            or not np.isfinite(width) or width <= 0):
        raise ValueError('Aligned finite FP32/FP64 geometry and positive band required')
    edge = segments[:, 1]-segments[:, 0]
    norm = np.sum(edge**2, axis=-1)
    if np.any(norm <= 0) or not np.isfinite(norm).all():
        raise ValueError('Finite positive segment lengths required')
    if not len(points):
        return np.zeros(0, dtype=bool)
    exact = segments.astype(np.float64)
    midpoint = (exact[:, 0]+exact[:, 1])/2
    half = np.linalg.norm(exact[:, 1]-exact[:, 0], axis=1)/2
    scale = max(float(np.max(np.abs(exact))), float(np.max(np.abs(points))), width, 1.)
    radius = half+width+64*np.finfo(points.dtype).eps*scale
    neighbors = cKDTree(points).query_ball_point(midpoint, radius, return_sorted=True)
    sizes = np.array([len(row) for row in neighbors])
    result = np.zeros(len(points), dtype=bool)
    if not sizes.sum():
        return result
    rows = np.concatenate(neighbors).astype(np.int64)
    segment_ids = np.repeat(np.arange(len(segments)), sizes)
    offset = points[rows]-segments[segment_ids, 0]
    selected_edge = edge[segment_ids]
    t = np.clip(np.sum(offset*selected_edge, axis=-1)/norm[segment_ids], 0., 1.)
    inside = np.sum((offset-t[:, None]*selected_edge)**2, axis=-1) <= width**2
    np.logical_or.at(result, rows, inside)
    return result
