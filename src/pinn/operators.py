"""Non-local operator L_delta for peridynamics-informed PINN training.

Provides build_D() to construct the operator on 1D/2D/3D point sets,
and _LDelta with condense() for Dirichlet volume constraints.
"""

import jax.numpy as jnp
import jax


def _tophat_tophat_kernel_value(delta, dim):
    """Top-hat kernel constant J0 for dimension dim."""
    if dim == 1:
        return 3.0 / delta**3
    elif dim == 2:
        return 8.0 / (jnp.pi * delta**4)
    elif dim == 3:
        return 15.0 / (2.0 * jnp.pi * delta**5)
    else:
        raise ValueError(f"Unsupported dimension: {dim}")


def _cubic_kernel_1d(dist, delta):
    """Cubic influence function: J(z) = (30/δ³)·(1 - |z|/δ)²."""
    z = jnp.abs(dist) / delta
    return (30.0 / delta**3) * (1.0 - z)**2


_KERNEL_BUILDERS = {
    "tophat": lambda dist, delta, dim: jnp.full_like(
        dist, _tophat_tophat_kernel_value(delta, dim)),
    "cubic": lambda dist, delta, dim: _cubic_kernel_1d(dist, delta),
}


# ── Public API ───────────────────────────────────────────────────────────────

def build_D(x, delta, grid_shape=None, kernel="cubic"):
    """Build the non-local operator L_delta on a point set.

    Args:
        x: (N, dim) coordinates in non-dimensional units.
        delta: non-dimensional horizon.
        grid_shape: (nx, ny, ...) for conv-based on large grids (tophat only).
        kernel: "cubic" (smooth, C¹) or "tophat" (discontinuous).

    Returns:
        Dense (N,N) jnp array for small N, or _LDelta for large grids.
    """
    N, dim = x.shape
    dV = 1.0 / N

    if grid_shape is not None:
        if kernel != "tophat":
            raise NotImplementedError(
                f"Kernel '{kernel}' not supported with grid_shape. Use 'tophat'.")
        return _LDelta.from_grid(
            x, delta, _tophat_tophat_kernel_value(delta, dim), dV, grid_shape)

    # Dense path
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

    return D


# ── LDelta operator (grid-based, conv-driven) ───────────────────────────────

class _LDelta:
    """Non-local operator on a uniform Cartesian grid.

    D @ u = J0*dV * (n_neighbors * u - conv(u, kernel))

    where the kernel has 1 at each valid neighbour offset and 0 at centre.
    Matvec uses jax.lax.conv — no Python loops, fully jit-able.
    No matrix stored.
    """
    _grid_shape: tuple
    _kernel: jnp.ndarray       # (kx,) or (kx, ky) or (kx, ky, kz)
    _n_neighbors: jnp.ndarray  # (N,) neighbour count per point
    _J0_dV: float

    def __init__(self):
        raise NotImplementedError("Use _LDelta.from_grid()")

    @classmethod
    def from_grid(cls, x, delta, J0, dV, grid_shape):
        dim = x.shape[1]
        shape = tuple(int(s) for s in grid_shape)
        assert len(shape) == dim
        dx = [(float(jnp.max(x[:, d]) - jnp.min(x[:, d])) / (shape[d] - 1))
              for d in range(dim)]
        max_di = [int(delta / dx[d]) + 1 for d in range(dim)]

        offsets = []
        _collect_offsets(dim, dx, delta, max_di, (), offsets)

        k_shape = tuple(2 * m + 1 for m in max_di)
        kernel = jnp.zeros(k_shape)
        centre = tuple(m for m in max_di)
        for off in offsets:
            idx = tuple(centre[d] + off[d] for d in range(dim))
            kernel = kernel.at[idx].set(1.0)

        dn = _conv_dims(dim)
        n_grid = jax.lax.conv_general_dilated(
            jnp.ones((1, *shape, 1)),
            kernel.reshape(*k_shape, 1, 1),
            window_strides=(1,) * dim,
            padding='SAME',
            dimension_numbers=dn,
        )[0, ..., 0]
        n_neighbors = n_grid.ravel()

        self = cls.__new__(cls)
        self._grid_shape = shape
        self._kernel = kernel
        self._n_neighbors = n_neighbors
        self._J0_dV = J0 * dV
        return self

    def __matmul__(self, u):
        """D @ u for u of shape (N,) or (N_t, N)."""
        shape = self._grid_shape
        kernel = self._kernel
        J0_dV = self._J0_dV
        dim = len(shape)
        dn = _conv_dims(dim)

        squeeze = (u.ndim == 1)
        if squeeze:
            u = u[None, :]  # (1, N)

        batch = u.shape[0]
        u_grid = u.reshape(batch, *shape, 1)

        neighbor_sum = jax.lax.conv_general_dilated(
            u_grid,
            kernel.reshape(*kernel.shape, 1, 1),
            window_strides=(1,) * dim,
            padding='SAME',
            dimension_numbers=dn,
        )[..., 0]  # (batch, *shape)

        neighbor_sum = neighbor_sum.reshape(batch, -1)  # (batch, N)
        result = J0_dV * (self._n_neighbors * u - neighbor_sum)

        return result[0] if squeeze else result

    def condense(self, mask_free, u_constr=None):
        """Return a new operator restricted to free DOFs.

        Constrained points are fixed to u_constr (zeros if None) and
        participate in D·u via the non-local integral.
        """
        return _CondensedLDelta(self, mask_free, u_constr)


class DWithBC:
    """Dense L_delta operator with Dirichlet BC built in.

    Before matvec, constrained positions are set to bc_values.
    The caller passes full-grid vectors; BC enforcement is transparent.
    """
    def __init__(self, D_full, mask_free, bc_values):
        self._D = D_full
        self._free = mask_free
        self._bc = bc_values

    def __matmul__(self, u):
        """D @ u where u is (N_all,) or (N_all, N_t)."""
        if u.ndim == 1:
            return self._D @ jnp.where(self._free, u, self._bc)
        return self._D @ jnp.where(self._free[:, None], u, self._bc[:, None])


def condense_dense(D, mask_free):
    """Condense a dense D matrix: restrict to free DOFs.

    Since D[i,i] already sums contributions from ALL neighbours (including
    constrained), the condensed operator is simply D[free, free].

    Args:
        D: (N, N) dense full operator.
        mask_free: bool array of length N.

    Returns:
        D_free: (N_free, N_free) dense matrix.
    """
    return D[mask_free, :][:, mask_free]


class _CondensedLDelta:
    """L_delta with Dirichlet BC condensed: only operates on free DOFs.

    Constrained positions are fixed to u_constr and participate in D·u
    via the non-local integral.
    """

    def __init__(self, parent, mask_free, u_constr=None):
        self._parent = parent
        self._free_idx = mask_free
        self._u_constr = u_constr  # (N,) or None for zeros

    def __matmul__(self, u_free):
        """D_free @ u_free, where u_free has shape (N_free,) or (N_t, N_free)."""
        N_total = len(self._free_idx)
        squeeze = (u_free.ndim == 1)
        if squeeze:
            u_free = u_free[None, :]
        batch = u_free.shape[0]

        if self._u_constr is not None:
            u_full = jnp.broadcast_to(self._u_constr, (batch, N_total))
        else:
            u_full = jnp.zeros((batch, N_total))
        u_full = u_full.at[:, self._free_idx].set(u_free)

        result_full = self._parent @ (u_full[0] if batch == 1 else u_full)
        if batch == 1:
            result_full = result_full[None, :]

        result = result_full[:, self._free_idx]
        return result[0] if squeeze else result


# ── Helpers ──────────────────────────────────────────────────────────────────

def _conv_dims(dim):
    """ConvDimensionNumbers for NHWC → HWIO → NHWC in `dim` spatial dims."""
    from jax.lax import ConvDimensionNumbers
    lhs = (0, dim + 1) + tuple(range(1, dim + 1))
    rhs = (dim + 1, dim) + tuple(range(dim))
    return ConvDimensionNumbers(lhs_spec=lhs, rhs_spec=rhs, out_spec=lhs)


def _collect_offsets(dim, dx, delta, max_di, prefix, out):
    """Recursively collect integer offset tuples within delta."""
    if len(prefix) == dim:
        if any(p != 0 for p in prefix):
            dist2 = sum((prefix[d] * dx[d])**2 for d in range(dim))
            if dist2 <= delta**2:
                out.append(prefix)
        return
    d = len(prefix)
    for k in range(-max_di[d], max_di[d] + 1):
        _collect_offsets(dim, dx, delta, max_di, prefix + (k,), out)


# ── Tests ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    delta = 0.1

    # ---- 1D dense ----
    for N in [101, 401, 1001]:
        x = jnp.linspace(0.0, 1.0, N)[:, None]
        D = build_D(x, delta)
        assert D.shape == (N, N)
        assert jnp.max(jnp.abs(jnp.sum(D, axis=1))) < 1e-3
        assert jnp.allclose(D, D.T, atol=1e-6)

        u = jnp.sin(jnp.pi * x[:, 0])
        nl = D @ u
        lap = jnp.pi**2 * u
        interior = slice(N // 5, -N // 5)
        err = nl[interior] - lap[interior]
        rel = float(jnp.sqrt(jnp.mean(err**2)) / jnp.sqrt(jnp.mean(lap[interior]**2)))
        print(f"1D dense: N={N}, delta={delta}, rel-L2={rel:.4f}")

    # ---- 2D dense ----
    n = 30
    x1d = jnp.linspace(0.0, 1.0, n)
    xv, yv = jnp.meshgrid(x1d, x1d)
    x2 = jnp.stack([xv.ravel(), yv.ravel()], axis=-1)
    D = build_D(x2, delta)
    assert D.shape == (n * n, n * n)
    assert jnp.max(jnp.abs(jnp.sum(D, axis=1))) < 1e-3
    assert jnp.allclose(D, D.T, atol=1e-6)

    u = jnp.sin(jnp.pi * x2[:, 0]) * jnp.sin(jnp.pi * x2[:, 1])
    nl = D @ u
    lap = 2.0 * jnp.pi**2 * u
    b = int(delta * n + 1)
    err = nl.reshape(n, n)[b:-b, b:-b] - lap.reshape(n, n)[b:-b, b:-b]
    lap_int = lap.reshape(n, n)[b:-b, b:-b]
    rel = float(jnp.sqrt(jnp.mean(err**2)) / jnp.sqrt(jnp.mean(lap_int**2)))
    print(f"2D dense: N={n*n}, delta={delta}, rel-L2={rel:.4f}")

    # ---- 2D LDelta ----
    for n in [100, 200, 400]:
        x1d = jnp.linspace(0.0, 1.0, n)
        xv, yv = jnp.meshgrid(x1d, x1d)
        x2 = jnp.stack([xv.ravel(), yv.ravel()], axis=-1)
        D = build_D(x2, delta, grid_shape=(n, n))
        u = jnp.sin(jnp.pi * x2[:, 0]) * jnp.sin(jnp.pi * x2[:, 1])
        nl = D @ u
        lap = 2.0 * jnp.pi**2 * u
        b = max(int(delta * n + 1), 1)
        err = nl.reshape(n, n)[b:-b, b:-b] - lap.reshape(n, n)[b:-b, b:-b]
        lap_int = lap.reshape(n, n)[b:-b, b:-b]
        rel = float(jnp.sqrt(jnp.mean(err**2)) / jnp.sqrt(jnp.mean(lap_int**2)))
        print(f"2D LDelta: {n}x{n}, delta={delta}, rel-L2={rel:.4f}")

    # ---- 1D tanh (phase-field interface) ----
    # FEM initial condition: phi(x) = 0.5·(1 - tanh(k·(x-0.5)))
    # -Δphi = -k²·sech²(k·(x-0.5))·tanh(k·(x-0.5))
    k = 29.23  # sqrt(omega_phi/(2*alpha_phi)) * L_c from FEM config
    print(f"\n--- 1D tanh interface (k={k}) ---")
    for N in [101, 401, 1001]:
        x = jnp.linspace(0.0, 1.0, N)[:, None]
        D = build_D(x, delta)
        t = k * (x[:, 0] - 0.5)
        u = 0.5 * (1.0 - jnp.tanh(t))
        sech2 = 1.0 / jnp.cosh(t)**2
        lap_u = -k**2 * sech2 * jnp.tanh(t)          # -Δ tanh
        nl = D @ u
        interior = slice(N // 5, -N // 5)
        err = nl[interior] - lap_u[interior]
        rel = float(jnp.sqrt(jnp.mean(err**2)) / jnp.sqrt(jnp.mean(lap_u[interior]**2)))
        print(f"1D tanh: N={N}, delta={delta}, rel-L2={rel:.4f}")

    # ---- JIT test ----
    jit_mv = jax.jit(lambda u: D @ u)
    nl2 = jit_mv(u)
    assert jnp.allclose(nl, nl2, atol=1e-5), "jit"
    print("jit matvec ✓")

    # ---- Condense test (dense) ----
    N = 101
    x = jnp.linspace(0.0, 1.0, N)[:, None]
    D = build_D(x, delta)
    mask_free = jnp.ones(N, dtype=bool)
    mask_free = mask_free.at[:10].set(False)
    mask_free = mask_free.at[-10:].set(False)
    D_free = condense_dense(D, mask_free)

    u = jnp.sin(jnp.pi * x[:, 0])
    u_free = u[mask_free]
    nl_free = D_free @ u_free
    u_full = jnp.zeros(N)
    u_full = u_full.at[mask_free].set(u_free)
    nl_full = D @ u_full
    assert jnp.allclose(nl_free, nl_full[mask_free], atol=1e-5), "condense dense"
    print("condense dense ✓")

    # ---- Condense test (LDelta) ----
    n = 100
    x1d = jnp.linspace(0.0, 1.0, n)
    xv, yv = jnp.meshgrid(x1d, x1d)
    x2 = jnp.stack([xv.ravel(), yv.ravel()], axis=-1)
    D = build_D(x2, delta, grid_shape=(n, n))
    mask = jnp.ones(n * n, dtype=bool)
    mask = mask.at[:n * 5].set(False)
    D_free = D.condense(mask)

    u = jnp.sin(jnp.pi * x2[:, 0]) * jnp.sin(jnp.pi * x2[:, 1])
    u_free = u[mask]
    nl_free = D_free @ u_free
    u_full = jnp.zeros(n * n)
    u_full = u_full.at[mask].set(u_free)
    nl_full = D @ u_full
    assert jnp.allclose(nl_free, nl_full[mask], atol=1e-4), "condense LDelta"
    print("condense LDelta ✓")

    print("All tests passed.")
