"""Local finite-element enrichment diagnostics for fixed-history damage."""

import numpy as np


def make_hat_basis(x_points, center_index, count):
    """Build contiguous nodal CG1 hat functions around one grid index."""
    x = np.asarray(x_points, dtype=np.float64)
    if x.ndim != 1 or len(x) < 3 or not np.all(np.isfinite(x)):
        raise ValueError("x_points must be a finite one-dimensional grid")
    if not np.all(np.diff(x) > 0.0):
        raise ValueError("x_points must be strictly increasing")
    if not np.allclose(np.diff(x), np.diff(x)[0], rtol=1e-5, atol=1e-8):
        raise ValueError("local hat enrichment requires a uniform grid")
    if isinstance(center_index, (bool, np.bool_)) or int(center_index) != center_index:
        raise ValueError("center_index must be an integer")
    if isinstance(count, (bool, np.bool_)) or int(count) != count:
        raise ValueError("count must be an integer")
    center_index = int(center_index)
    count = int(count)
    if count < 1 or count % 2 == 0:
        raise ValueError("count must be a positive odd integer")
    half = count // 2
    first = center_index - half
    last = center_index + half
    if first < 0 or last >= len(x):
        raise ValueError("hat enrichment patch must lie inside the grid")
    h = x[1] - x[0]
    centers = x[first:last + 1]
    basis = np.maximum(
        1.0 - np.abs(x[:, None] - centers[None, :]) / h,
        0.0,
    )
    return basis, np.arange(first, last + 1, dtype=np.int64)


def solve_local_damage_correction(
    operator,
    N_l,
    N_G,
    H_points,
    d_base,
    basis,
    residual_scaling,
):
    """Solve the least-squares correction of a fixed-history damage field."""
    history = np.asarray(H_points, dtype=np.float64)
    damage = np.asarray(d_base, dtype=np.float64)
    enrichment = np.asarray(basis, dtype=np.float64)
    if history.ndim != 1 or not np.all(np.isfinite(history)):
        raise ValueError("H_points must be a finite one-dimensional array")
    if np.any(history < 0.0):
        raise ValueError("H_points must be non-negative")
    if damage.shape != history.shape or not np.all(np.isfinite(damage)):
        raise ValueError("d_base must be finite and match H_points shape")
    if enrichment.ndim != 2 or enrichment.shape[0] != len(history):
        raise ValueError("basis must have shape (len(H_points), K)")
    if not np.all(np.isfinite(enrichment)):
        raise ValueError("basis must be finite")
    if not np.isfinite(N_l) or N_l <= 0.0:
        raise ValueError("N_l must be positive and finite")
    if not np.isfinite(N_G) or N_G <= 0.0:
        raise ValueError("N_G must be positive and finite")
    if residual_scaling not in ("none", "jacobi"):
        raise ValueError("residual_scaling must be 'none' or 'jacobi'")
    if operator.shape != (len(history), len(history)):
        raise ValueError("operator shape must match history shape")

    factor = 2.0 * N_l / N_G
    A = operator.copy()
    A = A * N_l**2
    A.setdiag(A.diagonal() + 1.0 + factor * history)
    residual = A @ damage - factor * history
    system_basis = A @ enrichment
    if residual_scaling == "jacobi":
        scale = 1.0 + N_l**2 * operator.diagonal() + factor * history
        residual = residual / scale
        system_basis = system_basis / scale[:, None]
    coefficients, _, _, _ = np.linalg.lstsq(
        system_basis,
        -residual,
        rcond=None,
    )
    corrected = damage + enrichment @ coefficients
    raw_residual = A @ corrected - factor * history
    raw_absolute = np.abs(raw_residual)
    scale = 1.0 + N_l**2 * operator.diagonal() + factor * history
    scaled_absolute = raw_absolute / scale
    return (
        corrected,
        coefficients,
        float(np.mean(raw_absolute)),
        float(np.max(raw_absolute)),
        float(np.mean(scaled_absolute)),
        float(np.max(scaled_absolute)),
    )
