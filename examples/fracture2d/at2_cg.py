"""Checked fixed-history volume-SPD AT2 CG, shared by direct diagnostics."""

import jax
import jax.numpy as jnp
from jax.scipy.sparse.linalg import cg
import numpy as np
from scipy.sparse import diags


def damage_solver(operator, volumes, gc0, gc1, *, tol, atol, maxiter):
    """Original volume-SPD AT2 solve; CG convergence is checked explicitly."""
    if (not np.isfinite(tol) or not 0 < tol < 1 or not np.isfinite(atol) or not 0 < atol
            or type(maxiter) is not int or maxiter < 1 or not np.all(np.isfinite([gc0, gc1]))
            or min(gc0, gc1) <= 0
            or volumes.shape != (operator.shape[0],) or not np.all(np.isfinite(volumes))
            or np.min(volumes) <= 0):
        raise ValueError("invalid fixed-H CG controls/volumes")
    V = jnp.asarray(volumes, jnp.float64)
    D = operator.to_scipy().tocsr()
    mass_D = diags(volumes) @ D
    skew = mass_D - mass_D.T
    if skew.nnz and np.max(np.abs(skew.data)) > 1e-10 * max(1., np.max(np.abs(mass_D.data))):
        raise ValueError("fixed-H CG requires volume-symmetric damage operator")
    diagonal_D = jnp.asarray(D.diagonal(), jnp.float64)
    # Global segmented scans agree as standalone matvecs but have produced
    # divergent nested CG on the current GPU/JAX runtime. Pack the same CSR
    # rows once and reduce within fixed rows, as in the validated damage PINN.
    if D.dtype != np.float64 or D.shape != (len(volumes), len(volumes)) or not np.all(np.isfinite(D.data)):
        raise ValueError("fixed-H CG requires a finite FP64 model-shaped CSR operator")
    off_diagonal = D.copy()
    off_diagonal.setdiag(0.)
    off_diagonal.eliminate_zeros()
    off_diagonal.sort_indices()
    counts = np.diff(off_diagonal.indptr)
    width = int(np.max(counts))
    columns = np.zeros((len(volumes), width), np.int32)
    weights = np.zeros((len(volumes), width), np.float64)
    for row in range(len(volumes)):
        start, end = off_diagonal.indptr[row:row + 2]
        columns[row, :end-start] = off_diagonal.indices[start:end]
        weights[row, :end-start] = off_diagonal.data[start:end]
    columns, weights = jnp.asarray(columns), jnp.asarray(weights)

    def damage_matvec(d):
        return diagonal_D*d + jnp.sum(weights*d[columns], axis=1)

    @jax.jit
    def solve(H, initial):
        diagonal = V * (gc0 + 2 * H + gc1 * diagonal_D)

        def matvec(d):
            return V * ((gc0 + 2 * H) * d + gc1 * damage_matvec(d))

        d, _ = cg(matvec, 2 * V * H, x0=initial, tol=tol, atol=atol,
                  maxiter=maxiter, M=lambda r: r / diagonal)
        return d, matvec(d) - 2 * V * H

    def checked(H, initial):
        if (H.shape != V.shape or initial.shape != V.shape
                or np.asarray(H).dtype != np.float64 or np.asarray(initial).dtype != np.float64
                or not np.all(np.isfinite(np.asarray(H)))
                or not np.all(np.isfinite(np.asarray(initial))) or float(jnp.min(H)) < 0):
            raise ValueError("fixed-H CG received invalid history/shape")
        d, r = solve(H, initial)
        norm = float(jnp.linalg.norm(r))
        limit = max(atol, tol * float(jnp.linalg.norm(2 * V * H)))
        if not np.all(np.isfinite(np.asarray(d))) or norm > limit * (1 + 1e-3):
            raise RuntimeError(f"GPU AT2 CG failed explicit residual: {norm} > {limit}")
        return d

    return checked
