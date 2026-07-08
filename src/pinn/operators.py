"""Non-local operator L_delta for peridynamics-informed PINN training.

Provides build_D() factory to construct DenseD or SparseD operators.
Both implement __matmul__ for (N,) and (N, N_t) inputs.
"""

import numpy as np
import jax.numpy as jnp
import jax


# ── Kernel helpers ──────────────────────────────────────────────────────────

# ── Kernel registry ──────────────────────────────────────────────────────────

def _tophat_kernel(dist, delta, dim):
    J0 = {1: 3.0, 2: 8.0 / jnp.pi, 3: 15.0 / (2.0 * jnp.pi)}[dim] / delta**(2 + dim)
    return jnp.where(jnp.abs(dist) <= delta, jnp.full_like(dist, J0), 0.0)


def _cubic_kernel(dist, delta, dim):
    J0 = {1: 30.0, 2: 120.0 / jnp.pi, 3: 315.0 / (2.0 * jnp.pi)}[dim] / delta**(2 + dim)
    z = jnp.abs(dist) / delta
    return J0 * jnp.where(z <= 1.0, (1.0 - z)**2, 0.0)


def _quartic_kernel(dist, delta, dim):
    J0 = {1: 105.0 / 8.0, 2: 48.0 / jnp.pi, 3: 945.0 / (16.0 * jnp.pi)}[dim] / delta**(2 + dim)
    z2 = (dist / delta)**2
    return J0 * jnp.where(z2 <= 1.0, (1.0 - z2)**2, 0.0)


def _wendland_c2_kernel(dist, delta, dim):
    """Wendland C² kernel: J(z) = J0·(1-z)⁴·(4z+1). C² smooth, compact support."""
    J0 = {1: 42.0, 2: 1008.0 / (5.0 * jnp.pi), 3: 315.0 / jnp.pi}[dim] / delta**(2 + dim)
    z = jnp.abs(dist) / delta
    return J0 * jnp.where(z <= 1.0, (1.0 - z)**4 * (4.0 * z + 1.0), 0.0)


_KERNEL_BUILDERS = {
    "tophat":   _tophat_kernel,
    "cubic":    _cubic_kernel,
    "quartic":  _quartic_kernel,
    "wendland": _wendland_c2_kernel,
}




# ── DenseD ──────────────────────────────────────────────────────────────────

class DenseD:
    """Dense (N,N) non-local operator matrix.  Best for N < ~2000."""

    _D: jnp.ndarray  # (N, N)

    def __init__(self, x, delta, kernel="cubic"):
        import numpy as np
        N, dim = x.shape
        x_np = np.array(x)
        ranges = x_np.max(axis=0) - x_np.min(axis=0)
        vol = float(np.prod(ranges))

        # Detect structured grid for better dV, fall back to uniform.
        # NOTE: hx*hy is interior midpoint quadrature, not strict nodal weights
        # (trapezoidal half-weights at edges / quarter-weights at corners).
        # TODO: trapezoidal boundary weights for rigorous nodal quadrature.
        if dim <= 3:
            unique_dims = [np.unique(np.round(x_np[:, d], decimals=10)) for d in range(dim)]
            n_per_dim = [len(u) for u in unique_dims]
            if np.prod(n_per_dim) == N:
                spacings = [(u[-1] - u[0]) / max(n - 1, 1) for u, n in zip(unique_dims, n_per_dim)]
                dV = float(np.prod(spacings))
            else:
                dV = vol / N
        else:
            dV = vol / N

        diff = x[:, None, :] - x[None, :, :]
        dist = jnp.sqrt(jnp.sum(diff**2, axis=-1))
        # Kernel enforces compact support at z=1; mask is efficiency only
        in_horizon = (dist <= delta) & ~jnp.eye(N, dtype=bool)

        if kernel not in _KERNEL_BUILDERS:
            raise ValueError(f"Unknown kernel: {kernel}. Options: {list(_KERNEL_BUILDERS)}")
        J_vals = _KERNEL_BUILDERS[kernel](dist, delta, dim)

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
        self._dim = dim
        ranges = x_np.max(axis=0) - x_np.min(axis=0)
        vol = np.prod(ranges)

        # Detect structured grid for better dV, fall back to uniform
        if dim <= 3:
            unique_dims = [np.unique(np.round(x_np[:, d], decimals=10)) for d in range(dim)]
            n_per_dim = [len(u) for u in unique_dims]
            if np.prod(n_per_dim) == N:
                spacings = [(u[-1] - u[0]) / max(n - 1, 1) for u, n in zip(unique_dims, n_per_dim)]
                dV = float(np.prod(spacings))
            else:
                dV = float(vol) / N
        else:
            dV = float(vol) / N

        # Detect structured grid — use grid-based neighbor search (O(N·k))
        # Fall back to O(N²) pairwise for unstructured point sets.
        row_idx, col_idx, r_ij = self._build_neighbors_grid(x_np, delta, dim)

        if kernel not in _KERNEL_BUILDERS:
            raise ValueError(f"Unknown kernel: {kernel}. Options: {list(_KERNEL_BUILDERS)}")
        J = np.array(_KERNEL_BUILDERS[kernel](jnp.array(r_ij), delta, dim))

        self._row_idx_np = row_idx.astype(np.int32)
        self._col_idx_np = col_idx.astype(np.int32)
        self._vals_np = (-J * dV).astype(np.float32)

        self._row_idx = jnp.array(self._row_idx_np, dtype=jnp.int32)
        self._col_idx = jnp.array(self._col_idx_np, dtype=jnp.int32)
        self._vals = jnp.array(self._vals_np)

        diag = np.bincount(row_idx, weights=J * dV, minlength=N)
        self._diag = jnp.array(diag)
        # Average out-degree: used by sub_sample to estimate K' analytically
        self._k_avg = len(row_idx) / N

    @staticmethod
    def _build_neighbors_grid(x_np, delta, dim):
        """O(N·k) grid-based neighbor search for structured meshes."""
        import numpy as np
        N = len(x_np)

        # Detect structured grid: unique coords in each dim
        ux = np.unique(np.round(x_np[:, 0], decimals=10))
        Nx = len(ux)
        if dim == 2:
            uy = np.unique(np.round(x_np[:, 1], decimals=10))
            Ny = len(uy)
            Nz = 1
            uz = np.array([0.0])
            if Nx * Ny != N:
                return SparseD._build_neighbors_pairwise(x_np, delta)
        elif dim == 3:
            uy = np.unique(np.round(x_np[:, 1], decimals=10))
            uz = np.unique(np.round(x_np[:, 2], decimals=10))
            Ny, Nz = len(uy), len(uz)
            if Nx * Ny * Nz != N:
                return SparseD._build_neighbors_pairwise(x_np, delta)
        else:
            Ny = 1
            Nz = 1
            uy = np.array([0.0])
            uz = np.array([0.0])
            if Nx != N:
                return SparseD._build_neighbors_pairwise(x_np, delta)

        dx = (ux[-1] - ux[0]) / max(Nx - 1, 1)
        dy = (uy[-1] - uy[0]) / max(Ny - 1, 1) if dim >= 2 else 1.0
        dz = (uz[-1] - uz[0]) / max(Nz - 1, 1) if dim >= 3 else 1.0
        md_x = max(int(delta / dx) + 1, 1)
        md_y = max(int(delta / dy) + 1, 1) if dim >= 2 else 0
        md_z = max(int(delta / dz) + 1, 1) if dim >= 3 else 0

        # Precompute all offset candidates within horizon
        offsets, dists = [], []
        for di in range(-md_y, md_y + 1):
            for dj in range(-md_x, md_x + 1):
                for dk in range(-md_z, md_z + 1):
                    if di == 0 and dj == 0 and dk == 0:
                        continue
                    d_sq = (di * dy)**2 + (dj * dx)**2 + (dk * dz)**2
                    if d_sq <= delta**2:
                        offsets.append((di, dj, dk))
                        dists.append(np.sqrt(d_sq))

        # Map point index k to grid coords.
        # Assumes C-order meshgrid ordering (z-fastest for 3D):
        #   meshgrid(ux, uy, uz, indexing='xy') → shape (Ny, Nx, Nz)
        if dim == 1:
            if not np.allclose(x_np.ravel(), ux, rtol=1e-6, atol=1e-6):
                raise ValueError(
                    "SparseD grid search requires sorted 1D point ordering. "
                    "Use _build_neighbors_pairwise for arbitrary point sets."
                )
        elif dim == 2:
            expected = np.stack(np.meshgrid(ux, uy, indexing='xy'), axis=-1).reshape(-1, dim)
            if not np.allclose(x_np, expected, rtol=1e-6, atol=1e-6):
                raise ValueError(
                    "SparseD grid search requires C-order meshgrid point ordering. "
                    "Use _build_neighbors_pairwise for arbitrary point sets."
                )
        else:  # dim == 3
            expected = np.stack(np.meshgrid(ux, uy, uz, indexing='xy'), axis=-1).reshape(-1, dim)
            if not np.allclose(x_np, expected, rtol=1e-6, atol=1e-6):
                raise ValueError(
                    "SparseD grid search requires C-order meshgrid point ordering "
                    "(z-fastest). Use _build_neighbors_pairwise for arbitrary point sets."
                )
        idx_k = np.arange(N).reshape(Ny, Nx, Nz)

        rows, cols, r_vals = [], [], []
        for (di, dj, dk), dist in zip(offsets, dists):
            iy_lo = max(0, -di); iy_hi = min(Ny, Ny - di)
            ix_lo = max(0, -dj); ix_hi = min(Nx, Nx - dj)
            iz_lo = max(0, -dk); iz_hi = min(Nz, Nz - dk)
            if iy_lo >= iy_hi or ix_lo >= ix_hi or iz_lo >= iz_hi:
                continue
            src = idx_k[iy_lo:iy_hi, ix_lo:ix_hi, iz_lo:iz_hi].ravel()
            dst = idx_k[(iy_lo + di):(iy_hi + di),
                        (ix_lo + dj):(ix_hi + dj),
                        (iz_lo + dk):(iz_hi + dk)].ravel()
            rows.append(src)
            cols.append(dst)
            r_vals.append(np.full(len(src), dist))

        if not rows:
            return SparseD._build_neighbors_pairwise(x_np, delta)
        row_idx = np.concatenate(rows)
        col_idx = np.concatenate(cols)
        r_ij = np.concatenate(r_vals)
        return row_idx.astype(np.int32), col_idx.astype(np.int32), r_ij

    @staticmethod
    def _build_neighbors_pairwise(x_np, delta):
        """O(N²) pairwise neighbor search (fallback for unstructured)."""
        import numpy as np
        diff = x_np[:, None, :] - x_np[None, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=-1))
        N = len(x_np)
        in_horizon = (dist <= delta) & ~np.eye(N, dtype=bool)
        row_idx, col_idx = np.where(in_horizon)
        r_ij = dist[row_idx, col_idx]
        return row_idx.astype(np.int32), col_idx.astype(np.int32), r_ij

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

    def sub_sample(self, seed, K):
        """Randomly sample K core points and their δ-neighbors.

        All index work done in numpy (CPU) to avoid eager GPU dispatch overhead.
        Returns JAX arrays with fixed padded shapes for JIT stability.

        Returns:
            local_idx: (N_pad,) global indices (zeros for pad)
            sub_data: (row, col, vals, diag) — padded to fixed size
            core_mask: (N_pad,) bool mask (True only for core)
            K_all: actual number of core ∪ halo points (before padding)
        """
        import numpy as np

        # Analytically estimate K' = expected |core ∪ halo| via coverage formula:
        #   E[K'] = N · (1 − (1 − k/N)^K),  k = avg neighbors per point.
        # 20% safety margin handles variance (std(K') << 0.2·E[K'] for large N).
        _k = self._k_avg
        _expected_Kprime = self._N * (1.0 - (1.0 - _k / self._N) ** K)
        N_pad = min(self._N, int(np.ceil(_expected_Kprime * 1.2)))

        # Sub-graph edges: each of N_pad nodes has k neighbors, each neighbor is
        # in sub-graph with prob ≈ N_pad/N  →  E[E'] ≈ N_pad · k · (N_pad/N).
        total_edges = len(self._row_idx_np)
        E_pad = min(total_edges,
                    int(np.ceil(N_pad * _k * (N_pad / self._N) * 1.2)))

        # --- all index work in numpy ---
        row_np = self._row_idx_np
        col_np = self._col_idx_np
        vals_np = self._vals_np

        rng = np.random.RandomState(seed % (2**31 - 1))
        core = rng.choice(self._N, K, replace=False)

        # Boolean mask: O(nnz) indexed access, much faster than np.isin hash lookup
        core_set = np.zeros(self._N, dtype=bool)
        core_set[core] = True
        in_core = core_set[row_np]
        halo = np.unique(col_np[in_core])
        all_idx = np.unique(np.concatenate([core, halo]))
        K_all = len(all_idx)

        # Use -1 sentinel so membership test is a simple comparison
        # (avoids two O(total_edges) np.isin hash-table lookups)
        g2l = np.full(self._N, -1, dtype=np.int32)
        g2l[all_idx] = np.arange(K_all)

        local_row = g2l[row_np]
        local_col = g2l[col_np]
        keep = (local_row >= 0) & (local_col >= 0)

        sub_row = local_row[keep]
        sub_col = local_col[keep]
        sub_vals = vals_np[keep]
        sub_diag = np.bincount(sub_row, weights=-sub_vals, minlength=K_all)

        core_local = g2l[core]
        core_mask = np.zeros(K_all, dtype=bool)
        core_mask[core_local] = True

        # Pad or truncate to fixed shapes (N_pad, E_pad) for JIT stability.
        # Truncation is extremely rare (K_all > N_pad requires >20% above expected).
        if K_all < N_pad:
            all_idx  = np.pad(all_idx,  (0, N_pad - K_all))
            sub_diag = np.pad(sub_diag, (0, N_pad - K_all))
            core_mask = np.pad(core_mask, (0, N_pad - K_all))
        elif K_all > N_pad:
            # Safety truncation: drop tail nodes (not in core_mask after g2l remap)
            all_idx   = all_idx[:N_pad]
            sub_diag  = sub_diag[:N_pad]
            core_mask = core_mask[:N_pad]
            keep = (sub_row < N_pad) & (sub_col < N_pad)
            sub_row, sub_col, sub_vals = sub_row[keep], sub_col[keep], sub_vals[keep]
        n_edges = len(sub_row)
        if n_edges < E_pad:
            sub_row  = np.pad(sub_row,  (0, E_pad - n_edges))
            sub_col  = np.pad(sub_col,  (0, E_pad - n_edges))
            sub_vals = np.pad(sub_vals, (0, E_pad - n_edges))
        elif n_edges > E_pad:
            sub_row, sub_col, sub_vals = sub_row[:E_pad], sub_col[:E_pad], sub_vals[:E_pad]

        return (jnp.array(all_idx),
                (jnp.array(sub_row), jnp.array(sub_col),
                 jnp.array(sub_vals), jnp.array(sub_diag)),
                jnp.array(core_mask),
                K_all)


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
