"""Finite-difference Laplacian (2D/3D) with CSR sub-sampling.

Drop-in replacement for the nonlocal SparseD operator; shares the same
sub_sample / build_subgraph_pool API.
"""

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx

from nonlocal_pinn.operators import _pad_entry


class FiniteDiff(eqx.Module):
    """2D (5-point) or 3D (7-point) finite-difference Laplacian.

    COO + CSR construction is done once in ``__init__`` (numpy CPU), then
    ``sub_sample`` and ``build_subgraph_pool`` work identically to SparseD.

    Parameters
    ----------
    x_base : jnp.ndarray  (N, dim)
        Point coordinates (used only to verify C-order meshgrid ordering).
    Nx, Ny : int
        Grid dimensions along x, y.
    Nz : int | None
        Grid dimension along z.  ``None`` means 2D.
    """
    _N: int = eqx.field(static=True)
    _row_idx: jnp.ndarray
    _col_idx: jnp.ndarray
    _vals: jnp.ndarray
    _diag: jnp.ndarray
    _nei_idx_np: np.ndarray
    _nei_w_np: np.ndarray
    _nei_mask_np: np.ndarray
    _k_max: int = eqx.field(static=True)

    def __init__(self, x_base, Nx, Ny, Nz=None):
        x_np = np.array(x_base)
        N = len(x_np)
        dim2 = Nz is None

        # Validate C-order meshgrid ordering
        if dim2:
            ux = np.unique(np.round(x_np[:, 0], decimals=10))
            uy = np.unique(np.round(x_np[:, 1], decimals=10))
            expected = np.stack(np.meshgrid(ux, uy, indexing='xy'),
                                axis=-1).reshape(-1, 2)
            if not np.allclose(x_np, expected, rtol=1e-6, atol=1e-6):
                raise ValueError("FiniteDiff requires C-order meshgrid points")
            idx_k = np.arange(N).reshape(Ny, Nx)
            offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # y, x
            shape3 = (Ny, Nx, 1)
            # Use Nx spacing
            h2 = float(1.0 / max(Nx - 1, 1))**2
        else:
            ux = np.unique(np.round(x_np[:, 0], decimals=10))
            uy = np.unique(np.round(x_np[:, 1], decimals=10))
            uz = np.unique(np.round(x_np[:, 2], decimals=10))
            expected = np.stack(np.meshgrid(ux, uy, uz, indexing='xy'),
                                axis=-1).reshape(-1, 3)
            if not np.allclose(x_np, expected, rtol=1e-6, atol=1e-6):
                raise ValueError("FiniteDiff requires C-order meshgrid points")
            idx_k = np.arange(N).reshape(Ny, Nx, Nz)
            offsets = [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0),
                       (0, 0, -1), (0, 0, 1)]  # y, x, z
            shape3 = (Ny, Nx, Nz)
            h2 = float(1.0 / max(Nx - 1, 1))**2  # assume isotropic

        rows, cols = [], []
        for off in offsets:
            di, dj = off[0], off[1]
            dk = off[2] if len(off) == 3 else 0
            iy_lo = max(0, -di); iy_hi = min(Ny, Ny - di)
            ix_lo = max(0, -dj); ix_hi = min(Nx, Nx - dj)
            if dim2:
                iz_lo, iz_hi = 0, 1
            else:
                iz_lo = max(0, -dk); iz_hi = min(Nz, Nz - dk)
            if iy_lo >= iy_hi or ix_lo >= ix_hi or iz_lo >= iz_hi:
                continue
            src = idx_k[iy_lo:iy_hi, ix_lo:ix_hi].ravel()
            dst = idx_k[(iy_lo + di):(iy_hi + di),
                        (ix_lo + dj):(ix_hi + dj)].ravel()
            if not dim2:
                src = idx_k[iy_lo:iy_hi, ix_lo:ix_hi, iz_lo:iz_hi].ravel()
                dst = idx_k[(iy_lo + di):(iy_hi + di),
                            (ix_lo + dj):(ix_hi + dj),
                            (iz_lo + dk):(iz_hi + dk)].ravel()
            rows.append(src)
            cols.append(dst)

        row_idx = np.concatenate(rows).astype(np.int32)
        col_idx = np.concatenate(cols).astype(np.int32)

        vals = np.full(len(row_idx), -1.0 / h2, dtype=np.float32)
        self._vals = jnp.array(vals)
        self._row_idx = jnp.array(row_idx, dtype=jnp.int32)
        self._col_idx = jnp.array(col_idx, dtype=jnp.int32)
        self._N = N

        w_pos = np.full(len(row_idx), 1.0 / h2, dtype=np.float32)
        diag = np.bincount(row_idx, weights=w_pos, minlength=N)
        self._diag = jnp.array(diag)

        # --- CSR (identical to SparseD) ---
        order = np.argsort(row_idx, kind='stable')
        row_s = row_idx[order]
        col_s = col_idx[order]
        w_s = w_pos[order]
        counts = np.bincount(row_s, minlength=N)
        self._k_max = int(counts.max())
        offsets_csr = np.zeros(N, dtype=np.int64)
        offsets_csr[1:] = np.cumsum(counts)[:-1]
        slot = np.arange(len(row_s)) - offsets_csr[row_s]
        self._nei_idx_np = np.full((N, self._k_max), 0, dtype=np.int32)
        self._nei_w_np = np.zeros((N, self._k_max), dtype=np.float32)
        self._nei_mask_np = np.zeros((N, self._k_max), dtype=bool)
        self._nei_idx_np[row_s, slot] = col_s
        self._nei_w_np[row_s, slot] = w_s
        self._nei_mask_np[row_s, slot] = True

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
        """CSR-based sub-graph — identical logic to SparseD.sub_sample."""
        rng = np.random.RandomState(seed % (2**31 - 1))
        core = rng.choice(self._N, K, replace=False)

        nei_idx = self._nei_idx_np[core]
        nei_w   = self._nei_w_np[core]
        nei_mask = self._nei_mask_np[core]

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
        sub_vals = -w  # negative for off-diagonal
        sub_diag = np.bincount(sub_row, weights=w, minlength=K_all)

        core_local = g2l[core]
        core_mask = np.zeros(K_all, dtype=bool)
        core_mask[core_local] = True

        return (all_idx, (sub_row, sub_col, sub_vals, sub_diag), core_mask, K_all)

    def build_subgraph_pool(self, K, n_pool):
        """Build a pool of uniformly padded sub-graphs — identical to SparseD."""
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
