"""P1 local plane-strain FEM for a separate fracture isolation experiment.

The weak form is integral g(d) sigma_0(u):epsilon(v). Material g is integrated
by degree-two triangle quadrature; grip cells use 1+k_res. The history driver
is solver.py's hybrid plane-strain Amor split, lumped-L2 projected from constant
cell strains onto material nodes. This is not a common constrained u/d energy.
No FEniCS or JAX imports are needed for host assembly/independent UFL checks.
"""

import numpy as np
from scipy.sparse import coo_matrix


QUADRATURE = np.array([[2/3, 1/6, 1/6], [1/6, 2/3, 1/6], [1/6, 1/6, 2/3]])


def triangle_data(coordinates, triangles, E, nu):
    if (coordinates.dtype != np.float64 or coordinates.ndim != 2
            or coordinates.shape[1] != 2 or not np.all(np.isfinite(coordinates))
            or triangles.dtype != np.int64 or triangles.ndim != 2
            or triangles.shape[1] != 3 or len(triangles) == 0
            or np.min(triangles) < 0 or np.max(triangles) >= len(coordinates)
            or not np.isfinite(E) or E <= 0 or not np.isfinite(nu) or not 0 <= nu < .5):
        raise ValueError("invalid P1 coordinates, triangle indices or plane-strain material")
    vertices = coordinates[triangles]
    affine = np.concatenate((np.ones((*vertices.shape[:2], 1)), vertices), axis=2)
    determinants = np.linalg.det(affine)
    if np.any(np.abs(determinants) < 1e-15):
        raise ValueError("degenerate P1 triangle")
    gradients = np.linalg.inv(affine)[:, 1:, :].transpose(0, 2, 1)
    areas = .5 * np.abs(determinants)
    B = np.zeros((len(triangles), 3, 6))
    B[:, 0, 0::2] = gradients[:, :, 0]
    B[:, 1, 1::2] = gradients[:, :, 1]
    B[:, 2, 0::2] = gradients[:, :, 1]
    B[:, 2, 1::2] = gradients[:, :, 0]
    mu = E / (2 * (1 + nu))
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    C = np.array([[lam+2*mu, lam, 0.], [lam, lam+2*mu, 0.], [0., 0., mu]])
    stiffness = areas[:, None, None] * np.einsum('eai,ab,ebj->eij', B, C, B)
    dofs = (2*triangles[:, :, None] + np.arange(2)).reshape(-1, 6)
    return areas, B, stiffness, dofs, lam, mu


def occurrence_rows(rows, size):
    counts = np.bincount(rows, minlength=size)
    if np.any(counts == 0):
        raise ValueError("P1 mesh has an unused degree of freedom")
    width = int(np.max(counts))
    order = np.argsort(rows, kind='stable')
    starts = np.concatenate(([0], np.cumsum(counts)))
    indices = np.zeros((size, width), np.int32)
    mask = np.zeros((size, width), bool)
    for row in range(size):
        count = counts[row]
        indices[row, :count] = order[starts[row]:starts[row+1]]
        mask[row, :count] = True
    return indices, mask


class LocalMechanics:
    def __init__(self, coordinates, triangles, n_material, n_material_cells,
                 volumes, *, E, nu, k_res, constrained, load_direction):
        if (type(n_material) is not int or not 0 < n_material <= len(coordinates)
                or type(n_material_cells) is not int or not 0 < n_material_cells <= len(triangles)
                or not np.isfinite(k_res) or not 0 < k_res < 1
                or volumes.shape != (n_material,) or volumes.dtype != np.float64
                or not np.all(np.isfinite(volumes)) or np.any(volumes <= 0)
                or constrained.dtype != np.int64 or constrained.ndim != 1
                or len(constrained) == 0 or np.min(constrained) < 0
                or np.max(constrained) >= 2*len(coordinates)
                or not np.array_equal(np.unique(constrained), constrained)
                or load_direction.shape != (2*len(coordinates),)
                or load_direction.dtype != np.float64 or not np.all(np.isfinite(load_direction))):
            raise ValueError("invalid material prefix, volumes, stiffness or boundary constraints")
        self.coordinates, self.triangles = coordinates, triangles
        self.n_material, self.n_material_cells = n_material, n_material_cells
        self.volumes, self.k_res = volumes, k_res
        self.areas, self.B, self.element_stiffness, self.dofs, self.lam, self.mu = triangle_data(
            coordinates, triangles, E, nu)
        if np.max(triangles[:n_material_cells]) >= n_material:
            raise ValueError("material cells must only contain material nodes")
        projection = coo_matrix((np.repeat(self.areas[:n_material_cells]/3, 3),
            (triangles[:n_material_cells].reshape(-1), np.repeat(np.arange(n_material_cells), 3))),
            shape=(n_material, n_material_cells)).tocsr()
        np.testing.assert_allclose(np.asarray(projection.sum(axis=1)).ravel(), volumes, rtol=1e-12, atol=1e-15)
        self.projection = projection
        self.constrained, self.load_direction = constrained, load_direction
        self.free = np.ones(2*len(coordinates), bool)
        self.free[constrained] = False
        if np.any(load_direction[self.free] != 0):
            raise ValueError("prescribed loading direction must vanish on free DOFs")
        self.rows = np.broadcast_to(self.dofs[:, :, None], self.element_stiffness.shape).reshape(-1)
        self.columns = np.broadcast_to(self.dofs[:, None, :], self.element_stiffness.shape).reshape(-1)

    def factors(self, damage):
        if (damage.shape != (self.n_material,) or damage.dtype != np.float64
                or not np.all(np.isfinite(damage)) or np.min(damage) < -1e-10
                or np.max(damage) > 1+1e-10):
            raise ValueError("invalid P1 material damage")
        factors = np.full(len(self.triangles), 1+self.k_res)
        values = damage[self.triangles[:self.n_material_cells]] @ QUADRATURE.T
        factors[:self.n_material_cells] = np.mean((1-values)**2, axis=1)+self.k_res
        return factors

    def matrix(self, damage):
        values = self.element_stiffness * self.factors(damage)[:, None, None]
        n = len(self.load_direction)
        return coo_matrix((values.reshape(-1), (self.rows, self.columns)), shape=(n, n)).tocsr()

    def driver(self, displacement):
        if (displacement.shape != self.load_direction.shape or displacement.dtype != np.float64
                or not np.all(np.isfinite(displacement))):
            raise ValueError("invalid P1 displacement")
        strain = np.einsum('eai,ei->ea', self.B[:self.n_material_cells],
                           displacement[self.dofs[:self.n_material_cells]])
        trace = strain[:, 0]+strain[:, 1]
        norm = strain[:, 0]**2+strain[:, 1]**2+.5*strain[:, 2]**2
        positive = .5*(self.lam+2*self.mu/3)*np.maximum(trace, 0)**2 + self.mu*(norm-trace**2/3)
        return np.asarray(self.projection @ positive)/self.volumes

    def gpu_solver(self, *, tol, atol, maxiter):
        """Checked Jacobi-CG of the standard FEM weak form, with fixed padded rows."""
        import jax
        import jax.numpy as jnp
        from jax.scipy.sparse.linalg import cg
        if (not jax.config.jax_enable_x64 or not np.isfinite(tol) or not 0 < tol < 1
                or not np.isfinite(atol) or atol <= 0 or type(maxiter) is not int or maxiter < 1):
            raise ValueError("local FEM CG requires x64 and explicit positive controls")
        indices, mask = occurrence_rows(self.rows, len(self.load_direction))
        columns = jnp.asarray(self.columns[indices], jnp.int32)
        cells = jnp.asarray((indices//36).astype(np.int32))
        weights = jnp.asarray(np.where(mask, self.element_stiffness.reshape(-1)[indices], 0.))
        diagonal_weights = jnp.where(columns == jnp.arange(len(self.load_direction))[:, None], weights, 0.)
        free, direction = jnp.asarray(self.free), jnp.asarray(self.load_direction)
        material_triangles = jnp.asarray(self.triangles[:self.n_material_cells], jnp.int32)
        quadrature = jnp.asarray(QUADRATURE)

        @jax.jit
        def solve(damage, load, initial):
            g_material = jnp.mean((1-damage[material_triangles] @ quadrature.T)**2, axis=1)+self.k_res
            factors = jnp.concatenate((g_material, jnp.full(len(self.triangles)-self.n_material_cells, 1+self.k_res)))
            coefficients = weights*factors[cells]
            diagonal = jnp.sum(diagonal_weights*factors[cells], axis=1)

            def matvec(q):
                return jnp.sum(coefficients*q[columns], axis=1)

            boundary = load*direction
            rhs = jnp.where(free, -matvec(boundary), 0.)
            def constrained_matvec(q):
                return jnp.where(free, matvec(jnp.where(free, q, 0.)), q)
            x, _ = cg(constrained_matvec, rhs, x0=jnp.where(free, initial-boundary, 0.),
                      tol=tol, atol=atol, maxiter=maxiter,
                      M=lambda r: r/jnp.where(free, diagonal, 1.))
            q = jnp.where(free, x, 0.)+boundary
            force = matvec(q)
            limit = jnp.maximum(atol, tol*jnp.linalg.norm(rhs))
            return q, jnp.linalg.norm(jnp.where(free, force, 0.)), limit, jnp.dot(force, direction)

        def checked(damage, load, initial):
            import time
            self.factors(np.asarray(damage))
            if (not np.isfinite(float(load)) or float(load) <= 0
                    or initial.shape != self.load_direction.shape
                    or np.asarray(initial).dtype != np.float64 or not np.all(np.isfinite(np.asarray(initial)))):
                raise ValueError("invalid local FEM solve load/initial field")
            start = time.perf_counter()
            q, norm, limit, reaction = solve(jnp.asarray(damage), jnp.asarray(load), jnp.asarray(initial))
            q.block_until_ready()
            norm, limit, reaction = float(norm), float(limit), float(reaction)
            if not np.isfinite([norm, limit, reaction]).all() or norm > limit*(1+1e-3):
                raise RuntimeError(f"local FEM CG failed explicit force residual: {norm} > {limit}")
            if not np.all(np.isfinite(np.asarray(q))):
                raise FloatingPointError("local FEM CG displacement is nonfinite")
            return q, reaction, dict(converged=True, free_residual_norm=norm,
                convergence_limit=limit, seconds=time.perf_counter()-start)
        return checked
