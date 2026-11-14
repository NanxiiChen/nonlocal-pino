"""Local-linear log-error trend and conditional pointwise bootstrap intervals.

LOWESS uses tricube distance weights with no robust residual reweighting
(statsmodels LOWESS it=0). Circular residual blocks retain short-range order.
These are approximate trend intervals, not independent-run uncertainty.
"""

import numpy as np


def local_linear_matrix(x, evaluation, fraction):
    x, evaluation = np.asarray(x, float), np.asarray(evaluation, float)
    if (x.ndim != 1 or evaluation.ndim != 1 or len(x) < 5
            or not np.isfinite(x).all() or not np.isfinite(evaluation).all()
            or np.any(np.diff(x) <= 0) or not 0 < fraction <= 1
            or evaluation.min() < x[0] or evaluation.max() > x[-1]):
        raise ValueError("Require finite sorted observations and in-range evaluation points")
    neighbors = int(fraction*len(x))
    if neighbors < 4:
        raise ValueError("Local-linear smoothing requires at least four neighbors")
    distance = x[None, :]-evaluation[:, None]
    radius = np.sort(np.abs(distance), axis=1)[:, neighbors-1]
    if np.any(radius <= 0):
        raise ValueError("Degenerate local bandwidth")
    scaled = distance/radius[:, None]
    weights = (1-np.minimum(np.abs(scaled), 1)**3)**3
    s0, s1, s2 = weights.sum(axis=1), (weights*scaled).sum(axis=1), (weights*scaled**2).sum(axis=1)
    denominator = s0*s2-s1**2
    if np.any(denominator <= 0):
        raise ValueError("Rank-deficient local-linear fit")
    matrix = weights*(s2[:, None]-s1[:, None]*scaled)/denominator[:, None]
    np.testing.assert_allclose(matrix.sum(axis=1), 1, rtol=0, atol=1e-12)
    np.testing.assert_allclose(matrix@x, evaluation, rtol=1e-12, atol=1e-10)
    return matrix


def smooth_interval(x, values, *, fraction, block_size, replicates, seed, grid_points):
    """Return log-LOWESS and centered basic circular-block residual bootstrap.

    For z=log10(error), fitted=m=S_x z, r=z-m-mean(z-m). With circular
    resamples r*, fit z*=m+r*. Center bootstrap fits at S_grid m to remove
    double-smoothing drift. Their deviations are d*=S_grid r*. The pointwise
    basic interval is [S_grid z-q97.5(d*), S_grid z-q2.5(d*)], then exponentiated.
    This does not correct unknown smoothing bias in the original fit.
    """
    x, values = np.asarray(x, float), np.asarray(values, float)
    if (values.shape != x.shape or not np.isfinite(values).all() or np.any(values <= 0)
            or not 1 <= block_size < len(x) or replicates < 100 or grid_points < len(x)):
        raise ValueError("Invalid positive observations or bootstrap specification")
    grid = np.linspace(x[0], x[-1], grid_points)
    sample_matrix = local_linear_matrix(x, x, fraction)
    grid_matrix = local_linear_matrix(x, grid, fraction)
    log_values = np.log10(values)
    fitted = sample_matrix@log_values
    residual = log_values-fitted
    residual -= residual.mean()
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(x), size=(replicates, int(np.ceil(len(x)/block_size))))
    indices = ((starts[..., None]+np.arange(block_size)) % len(x)).reshape(replicates, -1)[:, :len(x)]
    deviations = residual[indices]@grid_matrix.T
    lower_quantile, upper_quantile = np.quantile(deviations, [.025, .975], axis=0)
    trend = grid_matrix@log_values
    smooth, lower, upper = 10**trend, 10**(trend-upper_quantile), 10**(trend-lower_quantile)
    if not np.isfinite(np.stack((smooth, lower, upper))).all() or np.any(lower > upper):
        raise FloatingPointError("Invalid bootstrap interval")
    return dict(x=grid, smooth=smooth, lower=lower, upper=upper,
                residual_log10=residual, bootstrap_indices=indices)
