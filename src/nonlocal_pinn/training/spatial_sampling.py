"""CPU sampling utilities that preserve a declared discrete spatial measure.

No geometry/operator or neural field is modified here. All random choices use
the supplied Generator, so worker completion order cannot change the draws.
"""
import numpy as np


def stratified_indices(masses, count, rng):
    """One random inverse-CDF draw in each equal-probability interval.

The average of f[indices] is unbiased for sum(masses*f)/sum(masses).
Individual slots have different distributions; only their complete average
has this identity. Never drop or separately renormalize selected slots.
"""
    masses = np.asarray(masses, dtype=np.float64)
    if (masses.ndim != 1 or not len(masses) or not np.isfinite(masses).all()
            or np.any(masses < 0) or not np.isfinite(masses.sum()) or not masses.sum() > 0):
        raise ValueError('Sampling requires finite nonnegative masses with positive total')
    if type(count) is not int or count < 1:
        raise ValueError('Sampling count must be a positive integer')
    cdf = np.cumsum(masses / masses.sum())
    cdf[-1] = 1.
    positions = (np.arange(count) + rng.random(count)) / count
    return np.searchsorted(cdf, positions, side='right')


def polyline_band(points, segments, width):
    """Distance to explicit line segments; no nearest-vertex approximation."""
    points, segments = np.asarray(points), np.asarray(segments)
    if (points.ndim != 2 or points.shape[1] != 2 or segments.ndim != 3
            or segments.shape[1:] != (2, 2) or not len(segments)
            or not np.isfinite(points).all() or not np.isfinite(segments).all()
            or not np.isfinite(width) or width <= 0):
        raise ValueError('Finite 2D points, nonempty segments and positive band width required')
    edge = segments[:, 1] - segments[:, 0]
    norm = np.sum(edge**2, axis=1)
    if np.any(norm <= 0):
        raise ValueError('Boundary segments must have positive length')
    result = np.zeros(len(points), dtype=bool)
    for start in range(0, len(points), 256):
        offset = points[start:start+256, None] - segments[None, :, 0]
        t = np.clip(np.sum(offset*edge, axis=-1)/norm, 0., 1.)
        result[start:start+256] = np.any(np.sum((offset-t[..., None]*edge)**2, axis=-1) <= width**2, axis=1)
    return result
