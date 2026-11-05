"""Open midpoint quadrature on the five reflecting rectangular-wall segments."""
import numpy as np


def _segment_count(length, spacing):
    """Ceil a geometric ratio, snapping only floating-point integer roundoff."""
    ratio = length / spacing
    nearest = np.rint(ratio)
    tolerance = 32 * np.finfo(np.float64).eps * max(1., abs(ratio))
    if abs(ratio-nearest) <= tolerance:
        ratio = nearest
    return max(1, int(np.ceil(ratio)))


def neumann_quadrature(c):
    """Return float64 wall points, outward unit normals and arc-length weights.

    Coordinate y increases into the specimen: the reservoir mouth lies on y=0,
    and the bottom wall is y=height. Segment order is left, right, bottom,
    top-left and top-right. Each segment has ceil(length/h) midpoint nodes,
    where h is the minimum nodal spacing of the declared rectangular grid.
    Corners and reservoir-opening endpoints are deliberately not sampled.
    """
    length, height = float(c.geometry.length), float(c.geometry.height)
    center, half = float(c.geometry.pit_x), float(c.geometry.mouth_half_width)
    if (not np.isfinite((length, height, center, half)).all()
            or min(length, height, half) <= 0
            or not 0 < center-half < center+half < length):
        raise ValueError('Positive rectangle dimensions and a mouth strictly inside the top edge are required')
    for name in ('Nx', 'Ny'):
        value = c.sample[name]
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 2:
            raise ValueError(f'sample.{name} must be an integer >=2 for wall quadrature')
    spacing = min(length/(c.sample.Nx-1), height/(c.sample.Ny-1))
    segments = (
        ((0., 0.), (0., height), (-1., 0.)),
        ((length, 0.), (length, height), (1., 0.)),
        ((0., height), (length, height), (0., 1.)),
        ((0., 0.), (center-half, 0.), (0., -1.)),
        ((center+half, 0.), (length, 0.), (0., -1.)),
    )
    points, normals, weights = [], [], []
    for start, end, normal in segments:
        start, end = np.asarray(start, dtype=np.float64), np.asarray(end, dtype=np.float64)
        direction = end-start
        segment_length = float(np.linalg.norm(direction))
        count = _segment_count(segment_length, spacing)
        fractions = (np.arange(count, dtype=np.float64)+.5)/count
        points.append(start[None]+fractions[:, None]*direction[None])
        normals.append(np.broadcast_to(np.asarray(normal, dtype=np.float64), (count, 2)).copy())
        weights.append(np.full(count, segment_length/count, dtype=np.float64))
    return np.concatenate(points), np.concatenate(normals), np.concatenate(weights)
