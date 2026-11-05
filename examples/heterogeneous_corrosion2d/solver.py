"""CPU physical-field BDF reference independent of PINO derivatives/losses.

An explicit positive negative-Laplacian matrix selects either independent local
FV or the frozen native nonlocal operator. Variable mobility affects AC only.
"""
import time

import numpy as np
from scipy import sparse
from scipy.integrate import solve_ivp

from .problem import boundary_mask, initial_fields, nondim


def make_system(c, matrix, points, kappa):
    n = len(points)
    matrix = sparse.csr_matrix(matrix, dtype=np.float64)
    kappa = np.asarray(kappa, dtype=np.float64)
    if matrix.shape != (n, n) or not np.isfinite(matrix.data).all():
        raise ValueError('Reference operator must be finite and square on the spatial grid')
    if kappa.shape != (n,) or not np.isfinite(kappa).all() or np.min(kappa) <= 0:
        raise ValueError('Reference requires a positive finite mobility at every point')
    if np.max(np.abs(matrix@np.ones(n))) > 1e-8:
        raise ValueError('Reference positive Laplacian must annihilate constants')
    off_diagonal = matrix-sparse.diags(matrix.diagonal())
    if np.min(matrix.diagonal()) < 0 or (off_diagonal.nnz and np.max(off_diagonal.data) > 1e-12):
        raise ValueError('Reference positive Laplacian needs nonnegative diagonal and nonpositive off-diagonal entries')
    nch, n1, n2, n3, dc, cle = nondim(c)
    mouth = boundary_mask(c, points)
    if not np.any(mouth) or np.all(mouth):
        raise ValueError('Reference grid needs both reservoir-mouth and free nodes')
    free = ~mouth
    ids = np.flatnonzero(free)
    nf = len(ids)
    ff = matrix[ids][:, ids].tocsr()
    km = kappa[ids]
    initial = initial_fields(c, points)

    def unpack(state):
        fields = np.zeros((n, 2), dtype=np.float64)
        fields[free, 0], fields[free, 1] = state[:nf], state[nf:]
        return fields

    def rhs(t, state):
        phi, conc = unpack(state).T
        hp = 6*phi*(1-phi)
        gp = 2*phi*(1-phi)*(1-2*phi)
        q = conc-dc*(3*phi**2-2*phi**3)-cle
        return np.r_[(kappa*(-n3*(matrix@phi)-n2*gp+n1*q*dc*hp))[free],
                     (-nch*(matrix@q))[free]]

    def jacobian(t, state):
        phi, conc = unpack(state)[free].T
        hp, hpp = 6*phi*(1-phi), 6-12*phi
        q = conc-dc*(3*phi**2-2*phi**3)-cle
        diagonal = -n2*(12*phi**2-12*phi+2)+n1*dc*(q*hpp-dc*hp**2)
        return sparse.bmat([
            [sparse.diags(km)@(-n3*ff+sparse.diags(diagonal)), sparse.diags(km*n1*dc*hp)],
            [nch*dc*ff@sparse.diags(hp), -nch*ff]], format='csc')

    return rhs, jacobian, np.r_[initial[free, 0], initial[free, 1]], unpack


def solve(c, matrix, points, kappa, times, rtol, atol, max_step):
    times = np.asarray(times, dtype=np.float64)
    if (times.ndim != 1 or len(times) < 2 or times[0] != 0 or times[-1] != c.sample.t_max
            or not np.isfinite(times).all() or np.any(np.diff(times) <= 0)
            or not np.isfinite([rtol, atol, max_step]).all() or min(rtol, atol, max_step) <= 0):
        raise ValueError('Reference times must increase from0to sample.t_max and tolerances must be positive finite')
    rhs, jacobian, state, unpack = make_system(c, matrix, points, kappa)
    start = time.perf_counter()
    result = solve_ivp(rhs, (0, c.sample.t_max), state, method='BDF', jac=jacobian, t_eval=times,
                       rtol=rtol, atol=atol, max_step=max_step)
    if not result.success or not np.array_equal(result.t, times):
        raise RuntimeError(f'Reference BDF failed: {result.message}')
    fields = np.stack([unpack(state) for state in result.y.T])
    if not np.isfinite(fields).all() or fields[..., 0].min() < -1e-5 or fields[..., 0].max() > 1.00001:
        raise ValueError('Reference physical phase range or finite-value check failed')
    info = dict(seconds=time.perf_counter()-start, nfev=result.nfev, njev=result.njev,
        nlu=result.nlu, rtol=float(rtol), atol=float(atol), max_step=float(max_step),
        phi_min=float(fields[..., 0].min()), phi_max=float(fields[..., 0].max()),
        c_min=float(fields[..., 1].min()), c_max=float(fields[..., 1].max()))
    return fields, info


def activation_screen(c, output):
    """Qualify activation control using homogeneous mobility and D-doubling.

    Saves the physical phase/concentration fields for all four controls. This
    is a CPU-only discretized-reference diagnostic, never supervised training
    data. ``output`` must not contain files with the intended output names.
    """
    import json
    from pathlib import Path

    from ml_collections import ConfigDict

    from .problem import grid, local_matrix, quadrature

    destination = Path(output)
    paths = [destination/'activation_screen.json', destination/'activation_screen.npz']
    if any(path.exists() for path in paths):
        raise FileExistsError(f'Activation-screen outputs already exist in {destination}')
    _, yy, points = grid(c)
    weights, matrix = quadrature(c), local_matrix(c)
    times = np.linspace(0.0, c.sample.t_max, c.operator.reference_snapshots)
    records, arrays = {}, {}
    diffusion = float(c.physical.n_ch)
    low, high = float(c.material.minimum), float(c.material.maximum)
    for mobility, diffusivity in ((low, diffusion), (high, diffusion),
                                 (low, 2*diffusion), (high, 2*diffusion)):
        control = ConfigDict(c.to_dict())
        control.physical.n_ch = diffusivity
        fields, info = solve(control, matrix, points, np.full(len(points), mobility),
            times, c.reference.rtol, c.reference.atol, c.reference.max_step)
        area = (1-fields[..., 0])@weights
        dissolved = fields[-1, :, 0] < 0.5
        if not np.any(dissolved):
            raise RuntimeError('Activation screening lost the initial dissolved region')
        key = f'k{mobility:g}_D{diffusivity:g}'
        records[key] = dict(**info, initial_area=float(area[0]),
            final_area=float(area[-1]), area_advance=float(area[-1]-area[0]),
            depth=float(yy.ravel()[dissolved].max()))
        arrays[key] = fields
        print(key, json.dumps(records[key]), flush=True)
    destination.mkdir(parents=True, exist_ok=True)
    # Fields use (time, x-fast spatial point, [phi,c]); points and weights share
    # the identical spatial ordering. Both filenames are direct paths.
    with paths[1].open('xb') as handle:
        np.savez_compressed(handle, points=points, times=times, weights=weights,
                            config_json=json.dumps(c.to_dict()), **arrays)
    with paths[0].open('x') as handle:
        json.dump(records, handle, indent=2)
        handle.write('\n')
    return records


def main():
    import argparse
    import importlib

    parser = argparse.ArgumentParser(description='CPU local-FV activation-control qualification')
    parser.add_argument('--output', required=True, help='Explicit output directory; never overwrite existing results')
    parser.add_argument('--config', default='examples.heterogeneous_corrosion2d.configs.train_base',
                        help='Module exposing the explicit get_config() factory')
    args = parser.parse_args()
    config = importlib.import_module(args.config).get_config()
    activation_screen(config, args.output)


if __name__ == '__main__':
    main()
