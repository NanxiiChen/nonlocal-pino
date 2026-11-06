"""Independent physical-field KKS BDF solver on a supplied positive Laplacian.

For local reference, D=M_lumped^{-1}K from standard P1 FEM. A native nonlocal
matrix may be supplied explicitly for a separate direct-discretization control.
Both retain the same open-mouth Dirichlet values and no-flux weak/graph walls.
No network, derivative of network weights, or PINO residual is used here.
"""
import time

import numpy as np
from scipy import sparse
from scipy.integrate import solve_ivp


def initial_fields(c, points, mouth):
    points, mouth = np.asarray(points), np.asarray(mouth)
    if (points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all()
            or mouth.shape != (len(points),) or mouth.dtype != np.bool_):
        raise ValueError('Initial fields require finite (N,2) points and a boolean mouth mask')
    radius = np.hypot(points[:, 0]-c.geometry.pit_x, points[:, 1])
    phi = 0.5*(1+np.tanh(c.geometry.interface_k*(radius-c.geometry.pit_radius)))
    fields = np.stack((phi, c.physical.c_se*phi**2*(3-2*phi)), axis=-1)
    fields[mouth] = 0.0
    return fields


def make_system(c, diffusion_matrix, points, mouth):
    p = c.physical
    nch, n1, n2, n3, kappa = p.n_ch, p.n_ac1, p.n_ac2, p.n_ac3, p.kappa
    dc, cle = p.c_se-p.c_le, p.c_le
    if (not np.isfinite([nch, n1, n2, n3, kappa, dc, cle]).all()
            or min(nch, n1, n2, n3, kappa, dc, cle) <= 0):
        raise ValueError('KKS coefficients, homogeneous kinetics and phase concentrations must be positive finite')
    if not np.isclose(n2/(2*n3), c.geometry.interface_k**2, rtol=1e-12):
        raise ValueError('Initial diffuse-interface width differs from the AC free-energy coefficients')
    initial = initial_fields(c, points, mouth)
    mouth = np.asarray(mouth)
    n = len(points)
    matrix = sparse.csr_matrix(diffusion_matrix, dtype=np.float64)
    if matrix.shape != (n, n) or not np.isfinite(matrix.data).all():
        raise ValueError('Reference diffusion matrix must be finite and square on the mesh nodes')
    row_scale = max(float(np.max(np.asarray(abs(matrix).sum(axis=1)))), 1.0)
    if (np.min(matrix.diagonal()) < 0
            or np.max(np.abs(matrix@np.ones(n))) > 1e-11*row_scale):
        raise ValueError('Positive Laplacian must have nonnegative diagonal and annihilate constants')
    # Deliberately do not impose graph M-matrix signs on a valid FEM stiffness.
    free = ~mouth
    ids = np.flatnonzero(free)
    nf = len(ids)
    if not nf:
        raise ValueError('Reference needs at least one free physical node')
    ff = matrix[ids][:, ids].tocsr()

    def unpack(state):
        fields = np.zeros((n, 2), dtype=np.float64)
        fields[free, 0], fields[free, 1] = state[:nf], state[nf:]
        return fields

    def rhs(t, state):
        phi, conc = unpack(state).T
        hp = 6*phi*(1-phi)
        gp = 2*phi*(1-phi)*(1-2*phi)
        q = conc-dc*phi**2*(3-2*phi)-cle
        return np.r_[kappa*(-n3*(matrix@phi)-n2*gp+n1*dc*q*hp)[free],
                     (-nch*(matrix@q))[free]]

    def jacobian(t, state):
        phi, conc = unpack(state)[free].T
        hp, hpp = 6*phi*(1-phi), 6-12*phi
        q = conc-dc*phi**2*(3-2*phi)-cle
        diagonal = -n2*(12*phi**2-12*phi+2)+n1*dc*(q*hpp-dc*hp**2)
        return sparse.bmat([
            [kappa*(-n3*ff+sparse.diags(diagonal)), sparse.diags(kappa*n1*dc*hp)],
            [nch*dc*ff@sparse.diags(hp), -nch*ff]], format='csc')

    return rhs, jacobian, np.r_[initial[free, 0], initial[free, 1]], unpack


def solve(c, diffusion_matrix, points, mouth, times, rtol, atol, max_step):
    times = np.asarray(times, dtype=np.float64)
    if (times.ndim != 1 or len(times) < 2 or times[0] != 0
            or times[-1] != c.sample.t_max or not np.isfinite(times).all()
            or np.any(np.diff(times) <= 0) or not np.isfinite([rtol, atol, max_step]).all()
            or min(rtol, atol, max_step) <= 0):
        raise ValueError('Reference needs ordered times from0to sample.t_max and positive finite tolerances')
    rhs, jacobian, state, unpack = make_system(c, diffusion_matrix, points, mouth)
    start = time.perf_counter()
    result = solve_ivp(rhs, (0, c.sample.t_max), state, method='BDF', jac=jacobian,
                       t_eval=times, rtol=rtol, atol=atol, max_step=max_step)
    if not result.success or not np.array_equal(result.t, times):
        raise RuntimeError(f'Physical KKS reference integration failed: {result.message}')
    fields = np.stack([unpack(values) for values in result.y.T])
    if (not np.isfinite(fields).all() or fields[..., 0].min() < -1e-5
            or fields[..., 0].max() > 1.00001):
        raise ValueError('Reference phase range or finite-value qualification failed; no clipping is applied')
    info = dict(seconds=time.perf_counter()-start, nfev=result.nfev, njev=result.njev,
        nlu=result.nlu, rtol=float(rtol), atol=float(atol), max_step=float(max_step),
        phi_min=float(fields[..., 0].min()), phi_max=float(fields[..., 0].max()),
        c_min=float(fields[..., 1].min()), c_max=float(fields[..., 1].max()))
    return fields, info
