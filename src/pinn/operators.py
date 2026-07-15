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




# ── Structured-grid detection ───────────────────────────────────────────────

def _detect_structured_grid(x_np):
    """Detect a C-order meshgrid point set.

    Returns (grid_shape, spacings, dV) where grid_shape is (Ny, Nx[, Nz]),
    or (None, None, dV_uniform) for unstructured point sets.
    """
    N, dim = x_np.shape
    ranges = x_np.max(axis=0) - x_np.min(axis=0)
    vol = float(np.prod(ranges))
    if dim > 3:
        return None, None, vol / N

    unique_dims = [np.unique(np.round(x_np[:, d], decimals=10)) for d in range(dim)]
    n_per_dim = [len(u) for u in unique_dims]
    if np.prod(n_per_dim) != N:
        return None, None, vol / N

    spacings = [(u[-1] - u[0]) / max(n - 1, 1) for u, n in zip(unique_dims, n_per_dim)]
    dV = float(np.prod(spacings))

    # Verify C-order meshgrid ordering (y-major, z-fastest for 3D)
    if dim == 1:
        expected = unique_dims[0][:, None]
        grid_shape = (n_per_dim[0],)
    elif dim == 2:
        expected = np.stack(np.meshgrid(unique_dims[0], unique_dims[1],
                                        indexing='xy'), axis=-1).reshape(-1, dim)
        grid_shape = (n_per_dim[1], n_per_dim[0])          # (Ny, Nx)
    else:
        expected = np.stack(np.meshgrid(unique_dims[0], unique_dims[1],
                                        unique_dims[2], indexing='xy'),
                            axis=-1).reshape(-1, dim)
        grid_shape = (n_per_dim[1], n_per_dim[0], n_per_dim[2])  # (Ny, Nx, Nz)

    if not np.allclose(x_np, expected, rtol=1e-6, atol=1e-6):
        return None, None, dV
    return grid_shape, spacings, dV


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

        # Detect structured grid — use grid-based neighbor search (O(N·k)).
        # Fall back to KD-tree (O(N log N)) for unstructured point sets,
        # with O(N²) pairwise as last resort when scipy is unavailable.
        try:
            row_idx, col_idx, r_ij = self._build_neighbors_grid(x_np, delta, dim)
        except ValueError:
            try:
                row_idx, col_idx, r_ij = self._build_neighbors_kdtree(x_np, delta)
            except ImportError:
                row_idx, col_idx, r_ij = self._build_neighbors_pairwise(x_np, delta)

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

        # CSR neighbour lists for fast sub-sampling: O(K·k) gather instead of
        # O(nnz) full-edge scan.  Built once; sub_sample then runs in ~ms.
        # Stores positive J·dV weights so D[u][i] = Σ_j w·(u[i] - u[j]).
        w_pos = (J * dV).astype(np.float32)
        order = np.argsort(row_idx, kind='stable')
        row_s = row_idx[order]
        col_s = col_idx[order]
        w_s = w_pos[order]
        counts = np.bincount(row_s, minlength=N)
        self._k_max = int(counts.max())
        offsets = np.zeros(N, dtype=np.int64)
        offsets[1:] = np.cumsum(counts)[:-1]
        slot = np.arange(len(row_s)) - offsets[row_s]
        self._nei_idx_np = np.full((N, self._k_max), 0, dtype=np.int32)
        self._nei_w_np = np.zeros((N, self._k_max), dtype=np.float32)
        self._nei_mask_np = np.zeros((N, self._k_max), dtype=bool)
        self._nei_idx_np[row_s, slot] = col_s
        self._nei_w_np[row_s, slot] = w_s
        self._nei_mask_np[row_s, slot] = True

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
                raise ValueError("Not a structured 2D grid")
        elif dim == 3:
            uy = np.unique(np.round(x_np[:, 1], decimals=10))
            uz = np.unique(np.round(x_np[:, 2], decimals=10))
            Ny, Nz = len(uy), len(uz)
            if Nx * Ny * Nz != N:
                raise ValueError("Not a structured 3D grid")
        else:
            Ny = 1
            Nz = 1
            uy = np.array([0.0])
            uz = np.array([0.0])
            if Nx != N:
                raise ValueError("Not a structured 1D grid")

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
            raise RuntimeError(
                f"No neighbours found within δ={delta} on structured grid "
                f"(dx≈{dx:.4f}).  Increase δ or reduce N.")
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

    @staticmethod
    def _build_neighbors_kdtree(x_np, delta):
        """O(N log N) KD-tree radius search for arbitrary point sets."""
        import numpy as np
        from scipy.spatial import cKDTree
        tree = cKDTree(x_np)
        pairs = tree.query_ball_tree(tree, delta)
        rows, cols, r_vals = [], [], []
        for i, neighbors in enumerate(pairs):
            for j in neighbors:
                if i != j:
                    rows.append(i)
                    cols.append(j)
                    r_vals.append(np.linalg.norm(x_np[i] - x_np[j]))
        return (np.array(rows, dtype=np.int32), np.array(cols, dtype=np.int32),
                np.array(r_vals))

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
        """CSR-based sub-graph: randomly sample K core points + δ-neighbors.

        Returns raw (unpadded) arrays.  For padded sub-graph pools with
        uniform shapes use :meth:`build_subgraph_pool`.

        Returns:
            all_idx: (K_all,) global indices of core ∪ halo
            sub_data: (row, col, vals, diag) — COO edge arrays
            core_mask: (K_all,) bool mask (True for core nodes)
            K_all: actual number of core ∪ halo nodes
        """
        import numpy as np

        rng = np.random.RandomState(seed % (2**31 - 1))
        core = rng.choice(self._N, K, replace=False)

        nei_idx = self._nei_idx_np[core]    # (K, k_max)
        nei_w   = self._nei_w_np[core]      # (K, k_max)
        nei_mask = self._nei_mask_np[core]  # (K, k_max)

        halo = np.unique(nei_idx[nei_mask])
        all_idx = np.unique(np.concatenate([core, halo]))
        K_all = len(all_idx)

        g2l = np.full(self._N, -1, dtype=np.int32)
        g2l[all_idx] = np.arange(K_all)

        k_per_core = nei_mask.sum(axis=1)
        src_global = np.repeat(core, k_per_core)
        dst_global = nei_idx[nei_mask]
        w = nei_w[nei_mask]

        sub_row = g2l[src_global]
        sub_col = g2l[dst_global]
        sub_vals = -w
        sub_diag = np.bincount(sub_row, weights=w, minlength=K_all)

        core_local = g2l[core]
        core_mask = np.zeros(K_all, dtype=bool)
        core_mask[core_local] = True

        return (all_idx, (sub_row, sub_col, sub_vals, sub_diag), core_mask, K_all)

    def build_subgraph_pool(self, K, n_pool):
        """Build *n_pool* sub-graphs with uniform (tight) padding.

        All entries are padded to the exact maximum node count and edge count
        observed across *all* pool entries — no estimation, no margin.
        """
        raw_entries = [self.sub_sample(i, K) for i in range(n_pool)]
        max_K = max(int(r[3]) for r in raw_entries)
        max_E = max(len(r[1][0]) for r in raw_entries)

        pool = []
        for r in raw_entries:
            p = _pad_entry(r, max_K, max_E)
            pool.append((jnp.array(p[0]),
                         (jnp.array(p[1][0]), jnp.array(p[1][1]),
                          jnp.array(p[1][2]), jnp.array(p[1][3])),
                         jnp.array(p[2]),
                         r[3]))
        return pool


# ── ConvD ───────────────────────────────────────────────────────────────────

class ConvD:
    """Non-local operator on a structured grid as a stencil convolution.

    On a uniform meshgrid the kernel weights depend only on the integer
    offset, so  D @ u = diag·u − conv(u, W)  with a fixed (2m+1)^dim stencil.
    diag_i = Σ_{j in domain} W_j is precomputed as conv(ones, W), which makes
    boundary truncation exact (identical weights and sums to DenseD/SparseD;
    only float summation order differs).

    The convolution is pinned to Precision.HIGHEST so results stay exact
    FP32 regardless of the global matmul-precision flag.  Works in 1D/2D/3D;
    batched inputs (N, N_t) treat N_t as conv batch.
    """

    def __init__(self, x, delta, kernel="cubic"):
        x_np = np.asarray(x, dtype=np.float64)
        N, dim = x_np.shape
        self._N = N
        self._dim = dim

        grid_shape, spacings, dV = _detect_structured_grid(x_np)
        if grid_shape is None:
            raise ValueError(
                "ConvD requires a structured C-order meshgrid point set. "
                "Use method='sparse' for unstructured points.")
        self._grid_shape = tuple(int(s) for s in grid_shape)

        if kernel not in _KERNEL_BUILDERS:
            raise ValueError(f"Unknown kernel: {kernel}. Options: {list(_KERNEL_BUILDERS)}")

        # Grid spacing per *grid axis* (grid_shape is (Ny, Nx[, Nz]) while
        # spacings is per coordinate dim (dx, dy[, dz]))
        if dim == 1:
            h = [spacings[0]]
        elif dim == 2:
            h = [spacings[1], spacings[0]]          # (hy, hx)
        else:
            h = [spacings[1], spacings[0], spacings[2]]  # (hy, hx, hz)

        # Stencil: all integer offsets with 0 < |r| <= delta
        m = [max(int(delta / hi) + 1, 1) for hi in h]
        axes = [np.arange(-mi, mi + 1) for mi in m]
        mesh = np.meshgrid(*axes, indexing='ij')
        r = np.sqrt(sum((g * hi) ** 2 for g, hi in zip(mesh, h)))
        center = tuple(mi for mi in m)
        # Kernel builders enforce compact support (0 beyond z=1); the center
        # (self) point is excluded explicitly.
        J = np.array(_KERNEL_BUILDERS[kernel](jnp.array(r), delta, dim))
        J[r > delta] = 0.0
        J[center] = 0.0
        W = (J * dV).astype(np.float32)             # positive weights J·dV

        # Trim all-zero border planes so the stencil is tight
        nz = np.argwhere(W != 0)
        lo = nz.min(axis=0); hi_ = nz.max(axis=0) + 1
        # symmetric support ⇒ trimmed stencil stays centered
        W = W[tuple(slice(l, u) for l, u in zip(lo, hi_))]

        self._W = jnp.array(W)
        self._pad = [(s // 2, s // 2) for s in W.shape]

        # diag_i = Σ_{j in domain} W_(j-i)  =  conv(ones, W)
        ones = jnp.ones((1, 1) + self._grid_shape, dtype=jnp.float32)
        diag = self._conv(ones)[0, 0]
        self._diag = diag.reshape(-1)               # (N,) in C order

    def _conv(self, u_nchw):
        """SAME-padded correlation with the stencil; exact FP32."""
        W = self._W[None, None]                     # (O=1, I=1, *k)
        spatial = 'HWD'[:self._dim]
        dn = jax.lax.conv_dimension_numbers(
            u_nchw.shape, W.shape,
            ('NC' + spatial, 'OI' + spatial, 'NC' + spatial))
        return jax.lax.conv_general_dilated(
            u_nchw, W, window_strides=(1,) * self._dim,
            padding=self._pad, dimension_numbers=dn,
            precision=jax.lax.Precision.HIGHEST)

    def __matmul__(self, u):
        single = (u.ndim == 1)
        u2 = u[:, None] if single else u            # (N, N_t)
        N_t = u2.shape[1]
        # (N_t, 1, *grid): batch over time steps, single channel
        g = u2.T.reshape((N_t, 1) + self._grid_shape)
        conv = self._conv(g).reshape(N_t, self._N).T   # (N, N_t)
        out = self._diag[:, None] * u2 - conv
        return out[:, 0] if single else out

    @property
    def shape(self):
        return (self._N, self._N)


# ── Padding helpers (module-level, pure numpy) ────────────────────────────────

def _pad_entry(raw, N_pad, E_pad):
    """Pad (or truncate) a raw sub-graph entry to *N_pad* nodes and *E_pad*
    edges.  Truncation triggers only if a pool entry exceeds the pre-computed
    maxima, which is extremely unlikely with 5 % margin.
    """
    import numpy as np
    all_idx, (sub_row, sub_col, sub_vals, sub_diag), core_mask, K_all = raw

    if K_all < N_pad:
        all_idx = np.pad(all_idx, (0, N_pad - K_all))
        sub_diag = np.pad(sub_diag, (0, N_pad - K_all))
        core_mask = np.pad(core_mask, (0, N_pad - K_all))
    elif K_all > N_pad:
        all_idx = all_idx[:N_pad]
        sub_diag = sub_diag[:N_pad]
        core_mask = core_mask[:N_pad]
        keep = (sub_row < N_pad) & (sub_col < N_pad)
        sub_row, sub_col, sub_vals = sub_row[keep], sub_col[keep], sub_vals[keep]

    n_edges = len(sub_row)
    if n_edges < E_pad:
        sub_row = np.pad(sub_row, (0, E_pad - n_edges))
        sub_col = np.pad(sub_col, (0, E_pad - n_edges))
        sub_vals = np.pad(sub_vals, (0, E_pad - n_edges))
    elif n_edges > E_pad:
        sub_row, sub_col, sub_vals = sub_row[:E_pad], sub_col[:E_pad], sub_vals[:E_pad]

    return (all_idx, (sub_row, sub_col, sub_vals, sub_diag), core_mask, K_all)


# ── Factory ─────────────────────────────────────────────────────────────────

def build_D(x, delta, method=None, kernel="cubic"):
    """Build the non-local operator L_delta on a point set.

    Args:
        x: (N, dim) coordinates in non-dimensional units.
        delta: non-dimensional horizon.
        method: "dense", "sparse", "conv", or None (auto: dense if N < 2000
            else sparse).  "conv" evaluates D as a stencil convolution —
            exact same weights/sums as "sparse", faster on full structured
            grids (requires C-order meshgrid points; no sub-graph sampling).
        kernel: "cubic" (smooth, C¹) or "tophat" (discontinuous).

    Returns:
        DenseD, SparseD, or ConvD instance (all support D @ u for (N,) and
        (N, N_t) inputs).

    """
    N = x.shape[0]

    if method is None:
        method = "dense" if N < 2000 else "sparse"

    if method == "dense":
        return DenseD(x, delta, kernel)
    elif method == "sparse":
        return SparseD(x, delta, kernel)
    elif method == "conv":
        return ConvD(x, delta, kernel)
    else:
        raise ValueError(
            f"Unknown method: {method}. Use 'dense', 'sparse', 'conv', or None.")


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

    # ---- ConvD vs SparseD (structured grids, 1D/2D/3D) ----
    print("\n--- ConvD vs SparseD ---")
    conv_tests = [("1D", jnp.linspace(0., 1., 401)[:, None])]
    x1d = jnp.linspace(0., 1., 60)
    xv, yv = jnp.meshgrid(x1d, jnp.linspace(0., 0.5, 40), indexing='xy')
    conv_tests.append(("2D", jnp.stack([xv.ravel(), yv.ravel()], axis=-1)))
    g3 = jnp.linspace(0., 1., 24)
    Xv, Yv, Zv = jnp.meshgrid(g3, g3, g3, indexing='xy')
    conv_tests.append(("3D", jnp.stack([Xv.ravel(), Yv.ravel(), Zv.ravel()],
                                       axis=-1)))
    for label, x in conv_tests:
        u = jnp.sin(jnp.pi * x[:, 0])
        for d in range(1, x.shape[1]):
            u = u * jnp.sin((d + 1) * jnp.pi * x[:, d])
        D_s = build_D(x, delta, method="sparse")
        D_c = build_D(x, delta, method="conv")
        y_s, y_c = D_s @ u, D_c @ u
        # fp32 summation-order round-off only; natural error scale is the
        # intermediate sum magnitude (diag·u), not the cancelled output
        cond = float(jnp.max(D_s._diag * jnp.abs(u))) + 1e-30
        dev = float(jnp.max(jnp.abs(y_c - y_s)))
        assert dev / cond < 1e-6, f"ConvD {label}: dev/cond {dev/cond:.2e}"
        uB = jnp.column_stack([u, 2 * u, 0.5 * u])
        devB = float(jnp.max(jnp.abs((D_c @ uB) - (D_s @ uB))))
        assert devB / (2 * cond) < 1e-6, f"ConvD {label} batched"
        jc = jax.jit(lambda v: D_c @ v)
        assert jnp.allclose(jc(u), y_c, atol=1e-6)
        print(f"  {label} N={len(x)}: dev/cond {dev/cond:.2e} ✓")

    print("All tests passed.")
