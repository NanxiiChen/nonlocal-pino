"""Non-local operator L_delta for peridynamics-informed PINN training.

Provides build_D() factory to construct DenseD or SparseD operators.
Both implement __matmul__ for (N,) and (N, N_t) inputs.
"""

import jax.numpy as jnp
import jax


# ── Kernel helpers ──────────────────────────────────────────────────────────

def _tophat_kernel_value(delta, dim):
    if dim == 1:
        return 3.0 / delta**3
    elif dim == 2:
        return 8.0 / (jnp.pi * delta**4)
    elif dim == 3:
        return 15.0 / (2.0 * jnp.pi * delta**5)
    else:
        raise ValueError(f"Unsupported dimension: {dim}")


def _cubic_kernel_value(delta, dim):
    """Cubic kernel normalization constant J0 for dimension dim.

    1D: J(z) = (30/δ³)·(1 - |z|/δ)²
    2D: J(z) = (60/(πδ⁴))·(1 - |z|/δ)²
    """
    if dim == 1:
        return 30.0 / delta**3
    elif dim == 2:
        return 120.0 / (jnp.pi * delta**4)
    else:
        raise ValueError(f"Unsupported dimension: {dim}")


def _cubic_kernel(dist, delta, dim):
    """Cubic influence function: J(z) = J0·(1 - |z|/δ)²."""
    J0 = _cubic_kernel_value(delta, dim)
    z = jnp.abs(dist) / delta
    return J0 * (1.0 - z)**2


_KERNEL_BUILDERS = {
    "tophat": lambda dist, delta, dim: jnp.full_like(
        dist, _tophat_kernel_value(delta, dim)),
    "cubic": _cubic_kernel,
}


# ── DenseD ──────────────────────────────────────────────────────────────────

class DenseD:
    """Dense (N,N) non-local operator matrix.  Best for N < ~2000."""

    _D: jnp.ndarray  # (N, N)

    def __init__(self, x, delta, kernel="cubic"):
        N, dim = x.shape
        dV = float(jnp.prod(jnp.max(x, axis=0) - jnp.min(x, axis=0))) / N
        diff = x[:, None, :] - x[None, :, :]
        dist = jnp.sqrt(jnp.sum(diff**2, axis=-1))
        dx_est = (jnp.max(x) - jnp.min(x)) / (N - 1)
        in_horizon = (dist <= delta + 0.5 * float(dx_est)) & ~jnp.eye(N, dtype=bool)

        J_fn = _KERNEL_BUILDERS.get(kernel)
        if J_fn is None:
            raise ValueError(f"Unknown kernel: {kernel}. Options: {list(_KERNEL_BUILDERS)}")
        J_vals = J_fn(dist, delta, dim)

        D = jnp.zeros((N, N))
        D = D.at[in_horizon].set(-J_vals[in_horizon] * dV)
        D = D.at[jnp.diag_indices(N)].set(-jnp.sum(D, axis=1))
        self._D = D

    def __matmul__(self, u):
        return self._D @ u

    @property
    def shape(self):
        return self._D.shape

    @property
    def T(self):
        """Transpose (D is symmetric, returns self for compat)."""
        return self._D.T


# ── SparseD ─────────────────────────────────────────────────────────────────

class SparseD:
    """Non-local operator via neighbor table + scatter-add.

    O(N·k) memory, O(N·k) matvec for k avg neighbors.
    Supports arbitrary kernels on any point set.
    """

    _row_idx: jnp.ndarray   # (nnz,) source point index
    _col_idx: jnp.ndarray   # (nnz,) neighbor point index
    _vals: jnp.ndarray      # (nnz,) off-diagonal weight D[i,j]
    _diag: jnp.ndarray      # (N,) diagonal D[i,i]
    _N: int

    def __init__(self, x, delta, kernel="cubic"):
        import numpy as np
        x_np = np.array(x)
        N, dim = x_np.shape
        self._N = N
        vol = np.prod(x_np.max(axis=0) - x_np.min(axis=0))
        dV = float(vol) / N

        diff = x_np[:, None, :] - x_np[None, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=-1))
        dx_est = (x_np.max() - x_np.min()) / (N - 1)
        in_horizon = (dist <= delta + 0.5 * dx_est) & ~np.eye(N, dtype=bool)

        row_idx, col_idx = np.where(in_horizon)
        r_ij = dist[row_idx, col_idx]

        if kernel == "cubic":
            z = r_ij / delta
            J0 = {1: 30.0/delta**3, 2: 120.0/(np.pi*delta**4)}[dim]
            J = J0 * (1.0 - z)**2
        elif kernel == "tophat":
            J0 = {1: 3.0/delta**3, 2: 8.0/(np.pi*delta**4),
                  3: 15.0/(2.0*np.pi*delta**5)}[dim]
            J = np.full_like(r_ij, J0)
        else:
            raise ValueError(f"Unknown kernel: {kernel}")

        self._row_idx = jnp.array(row_idx, dtype=jnp.int32)
        self._col_idx = jnp.array(col_idx, dtype=jnp.int32)
        self._vals = jnp.array(-J * dV)

        diag = np.bincount(row_idx, weights=J * dV, minlength=N)
        self._diag = jnp.array(diag)

    def __matmul__(self, u):
        if u.ndim == 1:
            contrib = (jnp.zeros(self._N, dtype=u.dtype)
                       .at[self._row_idx]
                       .add(self._vals * u[self._col_idx]))
            return self._diag * u + contrib
        else:
            weighted = self._vals[:, None] * u[self._col_idx, :]
            contrib = jax.ops.segment_sum(weighted, self._row_idx,
                                           num_segments=self._N)
            return self._diag[:, None] * u + contrib

    @property
    def shape(self):
        return (self._N, self._N)


# ── Factory ─────────────────────────────────────────────────────────────────

def build_D(x, delta, method=None, kernel="cubic"):
    """Build the non-local operator L_delta on a point set.

    Args:
        x: (N, dim) coordinates in non-dimensional units.
        delta: non-dimensional horizon.
        method: "dense", "sparse", or None (auto: dense if N < 2000 else sparse).
        kernel: "cubic" (smooth, C¹) or "tophat" (discontinuous).

    Returns:
        DenseD or SparseD instance (both support D @ u for (N,) and (N, N_t)).
    """
    N = x.shape[0]

    if method is None:
        method = "dense" if N < 2000 else "sparse"

    if method == "dense":
        return DenseD(x, delta, kernel)
    elif method == "sparse":
        return SparseD(x, delta, kernel)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'dense', 'sparse', or None.")


# ══════════════════════════════════════════════════════════════════════════════
# Legacy code below — kept for reference, not used in current training.
# ══════════════════════════════════════════════════════════════════════════════

# ── GridD (conv-based, uniform grid, tophat only) ───────────────────────────

# def _conv_dims(dim):
#     """ConvDimensionNumbers for NHWC → HWIO → NHWC in `dim` spatial dims."""
#     from jax.lax import ConvDimensionNumbers
#     lhs = (0, dim + 1) + tuple(range(1, dim + 1))
#     rhs = (dim + 1, dim) + tuple(range(dim))
#     return ConvDimensionNumbers(lhs_spec=lhs, rhs_spec=rhs, out_spec=lhs)
#
#
# def _collect_offsets(dim, dx, delta, max_di, prefix, out):
#     """Recursively collect integer offset tuples within delta."""
#     if len(prefix) == dim:
#         if any(p != 0 for p in prefix):
#             dist2 = sum((prefix[d] * dx[d])**2 for d in range(dim))
#             if dist2 <= delta**2:
#                 out.append(prefix)
#         return
#     d = len(prefix)
#     for k in range(-max_di[d], max_di[d] + 1):
#         _collect_offsets(dim, dx, delta, max_di, prefix + (k,), out)
#
#
# class _GridD:
#     """Non-local operator on a uniform Cartesian grid (conv-based).
#
#     D @ u = J0*dV * (n_neighbors * u - conv(u, kernel))
#     Matvec uses jax.lax.conv — no Python loops, fully jit-able.
#     """
#     _grid_shape: tuple
#     _kernel: jnp.ndarray
#     _n_neighbors: jnp.ndarray
#     _J0_dV: float
#
#     def __init__(self):
#         raise NotImplementedError("Use _GridD.from_grid()")
#
#     @classmethod
#     def from_grid(cls, x, delta, J0, dV, grid_shape):
#         ...
#
#     def __matmul__(self, u):
#         ...
#
#     def condense(self, mask_free, u_constr=None):
#         return _CondensedGridD(self, mask_free, u_constr)
#
#
# class _CondensedGridD:
#     """_GridD with Dirichlet BC condensed: only operates on free DOFs."""
#     ...
#
#
# class DWithBC:
#     """Dense operator with Dirichlet BC built in."""
#     ...
#
#
# def condense_dense(D, mask_free):
#     """Condense a dense D matrix: restrict to free DOFs."""
#     return D[mask_free, :][:, mask_free]


# ── Tests ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    delta = 0.1

    # ---- 1D Laplacian ----
    for N in [101, 401, 1001]:
        x = jnp.linspace(0.0, 1.0, N)[:, None]
        D = build_D(x, delta)
        assert isinstance(D, DenseD if N < 2000 else SparseD)
        u = jnp.sin(jnp.pi * x[:, 0])
        nl = D @ u
        lap = jnp.pi**2 * u
        interior = slice(N // 5, -N // 5)
        err = nl[interior] - lap[interior]
        rel = float(jnp.sqrt(jnp.mean(err**2)) / jnp.sqrt(jnp.mean(lap[interior]**2)))
        print(f"1D Laplacian: N={N}, delta={delta}, rel-L2={rel:.4f}  ({type(D).__name__})")

    # ---- 2D Laplacian ----
    for n in [30, 50]:
        x1d = jnp.linspace(0.0, 1.0, n)
        xv, yv = jnp.meshgrid(x1d, x1d)
        x2 = jnp.stack([xv.ravel(), yv.ravel()], axis=-1)
        D = build_D(x2, delta)
        u = jnp.sin(jnp.pi * x2[:, 0]) * jnp.sin(jnp.pi * x2[:, 1])
        nl = D @ u
        lap = 2.0 * jnp.pi**2 * u
        b = int(delta * n + 1)
        e = nl.reshape(n, n)[b:-b, b:-b] - lap.reshape(n, n)[b:-b, b:-b]
        l = lap.reshape(n, n)[b:-b, b:-b]
        rel = float(jnp.sqrt(jnp.mean(e**2)) / jnp.sqrt(jnp.mean(l**2)))
        print(f"2D Laplacian: {n}x{n} N={n*n}, delta={delta}, rel-L2={rel:.4f}  ({type(D).__name__})")

    # ---- 1D tanh interface ----
    k = 29.23
    print(f"\n--- 1D tanh interface (k={k}) ---")
    for N in [101, 401, 1001]:
        x = jnp.linspace(0.0, 1.0, N)[:, None]
        D = build_D(x, delta)
        t = k * (x[:, 0] - 0.5)
        u = 0.5 * (1.0 - jnp.tanh(t))
        sech2 = 1.0 / jnp.cosh(t)**2
        lap_u = -k**2 * sech2 * jnp.tanh(t)
        nl = D @ u
        interior = slice(N // 5, -N // 5)
        err = nl[interior] - lap_u[interior]
        rel = float(jnp.sqrt(jnp.mean(err**2)) / jnp.sqrt(jnp.mean(lap_u[interior]**2)))
        print(f"1D tanh: N={N}, delta={delta}, rel-L2={rel:.4f}")

    # ---- Sparse vs dense consistency ----
    print("\n--- Sparse vs dense ---")
    tests = [
        ("1D cubic",  jnp.linspace(0., 1., 101)[:, None], "cubic"),
        ("1D cubic",  jnp.linspace(0., 1., 401)[:, None], "cubic"),
    ]
    for n, kernel in [(30, "cubic"), (50, "tophat")]:
        x1d = jnp.linspace(0., 1., n)
        xv, yv = jnp.meshgrid(x1d, jnp.linspace(0., 0.5, n))
        x2 = jnp.stack([xv.ravel(), yv.ravel()], axis=-1)
        tests.append((f"2D {kernel}", x2, kernel))

    for label, x, kernel in tests:
        u = jnp.sin(jnp.pi * x[:, 0])
        if x.shape[1] > 1:
            u = u * jnp.sin(2 * jnp.pi * x[:, 1])
        N = len(x)
        D_d = build_D(x, delta, method="dense", kernel=kernel)
        D_s = build_D(x, delta, method="sparse", kernel=kernel)
        nnz = len(D_s._vals) if hasattr(D_s, '_vals') else 0
        assert jnp.allclose(D_d @ u, D_s @ u, atol=5e-3), f"{label} N={N}"
        # Batched
        u3 = jnp.column_stack([u, 2*u, 3*u])
        for j in range(3):
            assert jnp.allclose(D_d @ u3[:, j], D_s @ u3[:, j], atol=1e-2), \
                f"{label} batched col{j} N={N}"
        # JIT
        jd = jax.jit(lambda v: D_d @ v)
        js = jax.jit(lambda v: D_s @ v)
        assert jnp.allclose(jd(u), js(u), atol=5e-3), f"{label} JIT N={N}"
        print(f"  {label} N={N} kernel={kernel} nnz={nnz} ✓")

    # ---- Auto method ----
    print("\n--- Auto method ---")
    D_small = build_D(jnp.linspace(0., 1., 500)[:, None], delta)
    D_large = build_D(jnp.linspace(0., 1., 2500)[:, None], delta)
    assert isinstance(D_small, DenseD)
    assert isinstance(D_large, SparseD)
    print(f"  N=500 → DenseD, N=2500 → SparseD ✓")

    # ---- JIT ----
    x_jit = jnp.linspace(0., 1., 201)[:, None]
    D_jit = build_D(x_jit, delta)
    u_jit = jnp.sin(jnp.pi * x_jit[:, 0])
    f = jax.jit(lambda v: D_jit @ v)
    assert jnp.allclose(f(u_jit), D_jit @ u_jit, atol=1e-5)
    print("jit matvec ✓")

    print("All tests passed.")
