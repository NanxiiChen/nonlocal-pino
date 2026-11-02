"""CH defect-correction diagnostic with externally frozen velocity, not full NSCH."""
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from examples.channel_nsch2d.problem import initial_field, material, wall_slopes


def difference_matrix(indices, weights, count):
    if indices.shape != weights.shape or indices.shape[0] != count:
        raise ValueError('Aligned fixed-size difference stencils required')
    rows = np.repeat(np.arange(count), indices.shape[1])
    matrix = sparse.coo_matrix((weights.ravel(), (rows, indices.ravel())),
                              shape=(count, count)).tocsr()
    return matrix - sparse.diags(np.asarray(matrix.sum(axis=1)).ravel())


class FrozenCH:
    def __init__(self, c, data, grid):
        self.c, self.data, self.grid = c, data, tuple(grid)
        self.points = data['points']
        self.n = len(self.points)
        if self.n != (grid[0] + 1)*(grid[1] + 1):
            raise ValueError('Mapped tensor node count does not match declared grid')
        self.Dx = difference_matrix(data['neighbors'], data['gradient_x'], self.n)
        self.Dy = difference_matrix(data['neighbors'], data['gradient_y'], self.n)
        self.L = difference_matrix(data['neighbors'], data['laplacian'], self.n)
        ix, iy = np.meshgrid(np.arange(grid[0] + 1), np.arange(grid[1] + 1), indexing='ij')
        wall = ((iy == 0) | (iy == grid[1])).ravel()
        self.inlet = (ix == 0).ravel()
        self.outlet = (ix == grid[0]).ravel() & ~wall
        self.interior = np.asarray(data['interior']) > 0
        self.natural = ~self.interior & ~self.inlet
        db, dt = wall_slopes(self.points[:, 0], data['amplitudes'], c, xp=np)
        nx, ny = np.zeros(self.n), np.zeros(self.n)
        bottom, top = (iy == 0).ravel(), (iy == grid[1]).ravel()
        nx[bottom], ny[bottom] = db[bottom], -1.
        nx[top], ny[top] = -dt[top], 1.
        nx[self.outlet] = 1.
        length = np.maximum(np.hypot(nx, ny), 1.)
        self.normal = np.column_stack((nx / length, ny / length))
        self.N = sparse.diags(self.normal[:, 0])@self.Dx + sparse.diags(self.normal[:, 1])@self.Dy
        self.I = sparse.eye(self.n, format='csr')
        self.S = sparse.diags(self.interior.astype(float))
        self.B = sparse.diags(self.inlet.astype(float)) + sparse.diags(self.natural.astype(float))@self.N
        self.scale = np.concatenate((np.where(self.interior, c.loss.phase_scale, c.loss.boundary_phi_scale),
                                     np.where(self.interior, c.loss.chemical_scale, c.loss.boundary_mu_scale)))

    def initial(self):
        phi = initial_field(self.points, self.c, xp=np)
        A, kappa = material(self.c)
        return np.concatenate((phi, A*(phi**3 - phi) - kappa*(self.L@phi)))

    def natural_data(self, field):
        if field.shape != (self.n, 2):
            raise ValueError('Natural data require node-aligned phi/mu channels')
        return np.column_stack((self.N@field[:, 0], self.N@field[:, 1]))

    def residual(self, z, history_phi, velocity, natural, alpha, dt, *, jacobian):
        if z.shape != (2*self.n,) or history_phi.shape != (self.n,) or velocity.shape != (self.n, 2) or natural.shape != (self.n, 2):
            raise ValueError('Frozen CH state/history/velocity/BC shapes differ')
        phi, mu = z[:self.n], z[self.n:]
        u, v = velocity.T
        A, kappa = material(self.c)
        phase = (alpha*phi - history_phi)/dt + self.Dx@(phi*u) + self.Dy@(phi*v) - self.c.physical.mobility*(self.L@mu)
        chemical = mu - A*(phi**3 - phi) + kappa*(self.L@phi)
        phase[~self.interior] = (self.B@phi - self.inlet - self.natural*natural[:, 0])[~self.interior]
        chemical[~self.interior] = (self.B@mu - self.natural*natural[:, 1])[~self.interior]
        residual = np.concatenate((phase, chemical))/self.scale
        if not jacobian:
            return residual
        transport = self.Dx@sparse.diags(u) + self.Dy@sparse.diags(v)
        matrix = sparse.bmat(((self.S@(alpha/dt*self.I + transport) + self.B,
                               -self.c.physical.mobility*self.S@self.L),
                              (self.S@(kappa*self.L - sparse.diags(A*(3*phi**2 - 1))),
                               self.S + self.B)), format='csc')
        return residual, sparse.diags(1/self.scale)@matrix

    def step(self, initial, history_phi, velocity, natural, alpha, dt, tolerance, maximum):
        z = initial.copy()
        for iteration in range(maximum + 1):
            r, matrix = self.residual(z, history_phi, velocity, natural, alpha, dt, jacobian=True)
            error = float(np.max(np.abs(r)))
            if not np.isfinite(error):
                raise FloatingPointError('Nonfinite frozen-CH residual')
            if error < tolerance:
                return z, dict(iterations=iteration, maximum_scaled_residual=error)
            if iteration == maximum:
                raise RuntimeError(f'Frozen CH Newton failed: {error}')
            delta = splu(matrix.tocsc()).solve(-r)
            if not np.isfinite(delta).all():
                raise FloatingPointError('Nonfinite frozen-CH Newton update')
            merit = np.linalg.norm(r)
            for j in range(12):
                factor = 2.**(-j)
                trial = z + factor*delta
                rr = self.residual(trial, history_phi, velocity, natural, alpha, dt, jacobian=False)
                if np.isfinite(rr).all() and np.linalg.norm(rr) < (1 - 1e-4*factor)*merit:
                    z = trial
                    break
            else:
                raise RuntimeError('Frozen-CH line search failed')
        raise AssertionError('Unreachable frozen-CH step')
