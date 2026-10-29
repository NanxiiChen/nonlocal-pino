"""Independent physical-field BDF reference with explicit snapshot times.

Accepts a SciPy negative-Laplacian matrix, including the independently assembled
local FD/Neumann-mirror matrix or recorded native graph. No neural-network
outputs, parameter derivatives or training residual functions enter the solver.
"""
import time

import numpy as np
from scipy import sparse
from scipy.integrate import solve_ivp

from examples.corrosion2d.residual_experiment import boundary_values, initial_fields, nondim


def make_system(c, matrix, points):
    nch, n1, n2, n3, dc, cle = nondim(c)
    zero, one, boundary = boundary_values(c, points)
    free = ~(zero | one)
    ids = np.flatnonzero(free)
    nf = len(ids)
    if matrix.shape != (len(points), len(points)) or not np.isfinite(matrix.data).all():
        raise ValueError('Reference matrix must be finite and match the spatial grid')
    ff = matrix[ids][:, ids].tocsr()
    initial = initial_fields(c, points).astype(np.float64)
    initial[~free] = boundary[~free]
    jacobian_calls = 0

    def unpack(state):
        fields = boundary.copy()
        fields[free, 0], fields[free, 1] = state[:nf], state[nf:]
        return fields

    def rhs(t, state):
        phi, conc = unpack(state).T
        hp = 6*phi*(1-phi)
        gp = 2*phi*(1-phi)*(1-2*phi)
        q = conc-dc*(3*phi**2-2*phi**3)-cle
        return np.r_[(-n3*(matrix@phi)-n2*gp+n1*q*dc*hp)[free],
                     (-nch*(matrix@q))[free]]

    def jacobian(t, state):
        nonlocal jacobian_calls
        jacobian_calls += 1
        if jacobian_calls % 25 == 0:
            print(f'Reference N={len(points)} Jacobian{jacobian_calls} t*={t:.6g}', flush=True)
        phi, conc = unpack(state)[free].T
        hp, hpp = 6*phi*(1-phi), 6-12*phi
        q = conc-dc*(3*phi**2-2*phi**3)-cle
        diagonal = -n2*(12*phi**2-12*phi+2)+n1*dc*(q*hpp-dc*hp**2)
        return sparse.bmat([
            [-n3*ff+sparse.diags(diagonal), sparse.diags(n1*dc*hp)],
            [nch*dc*ff@sparse.diags(hp), -nch*ff]], format='csc')

    return rhs, jacobian, np.r_[initial[free, 0], initial[free, 1]], unpack


def solve(c, matrix, points, times, rtol, atol, max_step):
    times = np.asarray(times)
    if (times.ndim != 1 or len(times) < 2 or times[0] != 0. or times[-1] != 1.
            or not np.isfinite(times).all() or np.any(np.diff(times) <= 0)
            or min(rtol, atol, max_step) <= 0):
        raise ValueError('Reference needs ordered finite times spanning[0,1] and positive tolerances')
    rhs, jacobian, initial, unpack = make_system(c, matrix, points)
    start = time.perf_counter()
    print(f'Reference start N={len(points)} rtol={rtol:g} atol={atol:g}', flush=True)
    result = solve_ivp(rhs, (0., 1.), initial, method='BDF', jac=jacobian,
                       t_eval=times, rtol=rtol, atol=atol, max_step=max_step)
    if not result.success or not np.array_equal(result.t, times):
        raise RuntimeError(f'Reference BDF failed: {result.message}')
    fields = np.stack([unpack(state) for state in result.y.T])
    phi, conc = fields[..., 0], fields[..., 1]
    if (not np.isfinite(fields).all() or phi.min() < -1e-5 or phi.max() > 1.00001):
        raise ValueError(f'Reference phase/finite violation: phi={phi.min()}..{phi.max()}')
    # The invariant interval of AC phase is not an invariant interval of c:
    # c_t=N_CH*Laplacian(c-dc*h(phi)-c_le) contains a phase-curvature source.
    # The operator's linear-liquid map deliberately leaves c unrestricted.
    info = dict(seconds=time.perf_counter()-start, nfev=result.nfev, njev=result.njev,
                nlu=result.nlu, rtol=rtol, atol=atol, max_step=max_step,
                minimum=float(fields.min()), maximum=float(fields.max()),
                phi_min=float(phi.min()), phi_max=float(phi.max()),
                c_min=float(conc.min()), c_max=float(conc.max()))
    return fields, info
