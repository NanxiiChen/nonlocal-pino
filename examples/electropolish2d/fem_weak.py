"""Conforming P1 weak diffusion on the existing rectangular query grid.

Each cell uses its southwest--northeast diagonal. With P1 test functions N_i,
K_ij = integral grad(N_i).grad(N_j) and M_i = integral N_i, D=M_lump^-1 K
approximates -Delta. Time and reaction use nodal lumped quadrature. Side-wall
homogeneous Neumann flux vanishes in integration by parts; top/bottom values
remain in complete free rows and are enforced by the existing Dirichlet losses.
No numerical solution or future reference field enters this assembly.
"""
import numpy as np
from scipy import sparse

from nonlocal_pinn.operators import SparseD


SPEC = dict(kind='p1_lumped', triangulation='southwest_northeast',
            reaction='nodal', residual_norm='mass_normalized_l1')


def validate_fem(c):
    if (c.nl.method != 'fem_weak' or 'fem' not in c.operator
            or c.operator.fem.to_dict() != SPEC
            or c.operator.reference_action != 'load'
            or 'method_control' not in c.operator
            or c.operator.method_control != 'rolling_derivative_pair'
            or c.sample.spatial.mode != 'rolling_pool'
            or c.loss.use_causal or 'temporal_query' in c.arch):
        raise ValueError('FEM control requires explicit P1/lumped/nodal L1, rolling method control and reference load')


def assemble_rectangle(points, nx, ny):
    """Assemble exact right-triangle element gradients, mass and connectivity."""
    x = np.asarray(points, dtype=np.float64)
    if (type(nx) is not int or type(ny) is not int or min(nx, ny) < 3
            or x.shape != (nx*ny, 2) or not np.isfinite(x).all()):
        raise ValueError('Finite rectangular Nx*Ny coordinates with Nx,Ny>=3 required')
    a = x.reshape(ny, nx, 2)
    if (not np.array_equal(a[..., 0], np.broadcast_to(a[0, :, 0], (ny, nx)))
            or not np.array_equal(a[..., 1], np.broadcast_to(a[:, 0, 1, None], (ny, nx)))
            or np.any(np.diff(a[0, :, 0]) <= 0) or np.any(np.diff(a[:, 0, 1]) <= 0)):
        raise ValueError('Expected ascending Cartesian xy meshgrid, x varying fastest')
    nodes = np.arange(nx*ny).reshape(ny, nx)
    sw, se = nodes[:-1, :-1].ravel(), nodes[:-1, 1:].ravel()
    nw, ne = nodes[1:, :-1].ravel(), nodes[1:, 1:].ravel()
    triangles = np.concatenate((np.stack((sw, se, ne), axis=1), np.stack((sw, ne, nw), axis=1)))
    hx, hy = (x[se, 0]-x[sw, 0]), (x[nw, 1]-x[sw, 1])
    area = np.tile(hx*hy/2, 2)
    g1 = np.zeros((len(sw), 3, 2)); g2 = np.zeros_like(g1)
    g1[:, 0, 0] = -1/hx
    g1[:, 1, 0], g1[:, 1, 1], g1[:, 2, 1] = 1/hx, -1/hy, 1/hy
    g2[:, 0, 1], g2[:, 1, 0] = -1/hy, 1/hx
    g2[:, 2, 0], g2[:, 2, 1] = -1/hx, 1/hy
    gradients = np.concatenate((g1, g2))
    element = area[:, None, None]*np.einsum('eik,ejk->eij', gradients, gradients)
    row = np.broadcast_to(triangles[:, :, None], element.shape).ravel()
    col = np.broadcast_to(triangles[:, None, :], element.shape).ravel()
    stiffness = sparse.coo_matrix((element.ravel(), (row, col)), shape=(nx*ny, nx*ny)).tocsr()
    stiffness.sum_duplicates(); stiffness.eliminate_zeros()
    mass = np.bincount(triangles.ravel(), weights=np.repeat(area/3, 3), minlength=nx*ny)
    scale = max(1., np.max(np.abs(stiffness.data)))
    symmetry = stiffness-stiffness.T
    if (np.any(mass <= 0) or not np.isfinite(mass).all()
            or not np.isclose(mass.sum(), np.ptp(x[:, 0])*np.ptp(x[:, 1]), rtol=1e-13)
            or (symmetry.nnz and np.max(np.abs(symmetry.data)) > 1e-13*scale)
            or np.max(np.abs(stiffness@np.ones(len(x)))) > 1e-12*scale):
        raise ValueError('Positive integrating mass and symmetric constant-null stiffness required')
    return stiffness, mass, triangles


class RectangleWeakRows:
    """Complete sampled rows in the existing scalar core/halo payload format.

The generic SparseD row sampler is reused only for index selection/padding;
every coefficient comes from finite-element assembly, not a nonlocal kernel.
"""
    def __init__(self, points, nx, ny):
        stiffness, self.mass, self.triangles = assemble_rectangle(points, nx, ny)
        self.stiffness = stiffness
        self.matrix = (sparse.diags(1/self.mass)@stiffness).tocsr()
        self._N = len(self.mass)
        off = self.matrix.copy(); off.setdiag(0); off.eliminate_zeros()
        if np.any(off.data > 0) or np.any(self.matrix.diagonal() <= 0):
            raise ValueError('This right-triangle mesh must have nonpositive off-diagonals; no clipping allowed')
        coo = off.tocoo()
        self._row_idx_np, self._col_idx_np = coo.row.astype(np.int32), coo.col.astype(np.int32)
        self._vals_np = coo.data.astype(np.float32)
        self._diag = self.matrix.diagonal().astype(np.float32)
        counts = np.diff(off.indptr)
        self._k_max = int(counts.max())
        self._nei_idx_np = np.zeros((self._N, self._k_max), dtype=np.int32)
        self._nei_w_np = np.zeros((self._N, self._k_max), dtype=np.float32)
        self._nei_mask_np = np.zeros((self._N, self._k_max), dtype=bool)
        slots = np.arange(off.nnz)-np.repeat(off.indptr[:-1], counts)
        self._nei_idx_np[self._row_idx_np, slots] = self._col_idx_np
        self._nei_w_np[self._row_idx_np, slots] = -self._vals_np
        self._nei_mask_np[self._row_idx_np, slots] = True

    def sub_sample(self, seed, count):
        return SparseD.sub_sample(self, seed, count)

    def report(self):
        return dict(**SPEC, nodes=self._N, triangles=len(self.triangles),
                    mass_sum=float(self.mass.sum()), mass_min=float(self.mass.min()),
                    mass_max=float(self.mass.max()), max_neighbors=self._k_max,
                    directed_edges=len(self._vals_np),
                    boundary='Natural zero normal phi and q flux at left/right; soft top/bottom Dirichlet',
                    sampling='Uniform original core draws with all incident FE row couplings',
                    reference='Existing native graph is monitoring only; headline evaluation uses common local FD64')
